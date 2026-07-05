import os
import numpy as np
import nibabel as nib
import pyvista as pv
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from models.TransBraTS.TransBraTS_skipconnection import TransBraTS,IDH_ATRX_p19q_type_network
#from CBAM.model_resnet import ResidualNet3D
from CBAM.xiugai import ResidualNet3D
from models import criterions
from models.criterions import MultiTaskLossWrapper
from data.BraTS_IDH import BraTS
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import confusion_matrix
import logging
from torch.utils.tensorboard import SummaryWriter

class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        # self.cbam3d_model = ResidualNet3D(18, 6, 'CBAM3D')  # 这儿的num_classes随便写，因为后面resnet全连接层用的部分注释了，用不到
        self.cbam3d_model = ResidualNet3D(18,1000,'CBAM3D')
        # cbam3d_model2 = ResidualNet3D(18, 1000, 'CBAM3D')
        self.IDH_model = IDH_ATRX_p19q_type_network()
        idh_criterion = getattr(criterions, 'idh_lmfloss')  # idh_focal_loss, idh_cross_entropy,idh_lmfloss
        atrx_criterion = getattr(criterions, 'atrx_lmfloss')
        p19q_criterion = getattr(criterions, 'p19q_lmfloss')
        # criterion = FocalLoss_seg()
        self.MTL = MultiTaskLossWrapper(3, loss_fn=[idh_criterion, atrx_criterion, p19q_criterion])
        self.config = config

        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', config.experiment + config.date+"test")
        log_file = log_dir + '.txt'
        self.log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(config):
            logging.info('{}={}'.format(arg, getattr(config, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(config.description))

        self.best_epoch = 0
        self.min_loss = 100.0
        self.best_acc = 0
        self.best_acc_1 = 0
        self.epoch = 1

        self.checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                      config.experiment + config.date+"test")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        writer = SummaryWriter()

        self.optimizer =None

        resume = ''
        if os.path.isfile(resume) and config.load:
            logging.info('loading checkpoint {}'.format(resume))
            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['en_state_dict'])
            self.cbam3d_model.load_state_dict(checkpoint['cbam_state_dict'])
            # DDP_model['cbam2'].load_state_dict(checkpoint['cbam2_state_dict'])
            self.IDH_model.load_state_dict(checkpoint['idh_state_dict'])
            logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                         .format(config.resume, config.start_epoch))
        else:
            logging.info('re-training!!!')

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            amsgrad=self.config.amsgrad
        )
        return self.optimizer

    def on_train_epoch_start(self):
        self.epoch_train_logs = []


    def step(self, data):
        weight_IDH = torch.tensor([101, 67]).float()
        weight_ATRX = torch.tensor([50, 118]).float()
        weight_p19q = torch.tensor([40, 150]).float()
        x, idh, atrx, p19q, x_2,path = data
        layer4=self.cbam3d_model(x_2, 1)
        x1_1, x2_1, x3_2, x4_1, encoder_output, y,y1 = self.model(x,layer4)
        z=y
        y=layer4+y
        # #三维特征图的可视化
        # for i in range(128):
        #     selected_channel = x4_1[0, i, :, :, :].cpu()
        #     features=selected_channel.detach()
        #     # 假设features是一个PyTorch tensor，需要先转换为numpy数组并调整形状
        #     features_np = features.numpy()
        #
        #     # 确保数据是C-order（即连续存储），这是PyVista所期望的
        #     if not features_np.flags['C_CONTIGUOUS']:
        #         features_np = features_np.copy(order='C')
        #
        #     # # 使用ImageData代替UniformGrid
        #     # image_data = pv.ImageData()
        #     # image_data.dimensions = (features.shape[0], features.shape[1], features.shape[2])  # 注意这里的顺序
        #     # image_data.point_data["values"] = features_np.flatten(order="F")
        #     #
        #     # # 体积渲染
        #     # plotter = pv.Plotter()
        #     # plotter.add_volume(image_data, cmap="viridis", clim=[-1.0,3.0])
        #     # plotter.show()
        #     # 计算每个子块的维度
        #     dim_x, dim_y, dim_z = features.shape
        #     sub_dim_x = dim_x // 3
        #     sub_dim_y = dim_y // 3
        #     sub_dim_z = dim_z // 3
        #
        #     # 创建一个Plotter实例
        #     plotter = pv.Plotter(shape=(3, 3))  # 创建一个3x3的网格布局
        #
        #     # 循环遍历并分割特征图
        #     for z in range(3):
        #         for y in range(3):
        #             for x in range(3):
        #                 # 计算每个子块的范围
        #                 start_x = x * sub_dim_x
        #                 end_x = (x + 1) * sub_dim_x
        #                 start_y = y * sub_dim_y
        #                 end_y = (y + 1) * sub_dim_y
        #                 start_z = z * sub_dim_z
        #                 end_z = (z + 1) * sub_dim_z
        #
        #                 # 提取子块
        #                 sub_features = features_np[start_x:end_x, start_y:end_y, start_z:end_z]
        #
        #                 # 创建ImageData对象
        #                 image_data = pv.ImageData()
        #                 image_data.dimensions = (sub_dim_x, sub_dim_y, sub_dim_z)
        #                 image_data.point_data["values"] = sub_features.flatten(order="F")
        #
        #                 # 添加到Plotter
        #                 plotter.subplot(z, y)  # 注意这里使用y作为第二参数，因为plotter是从左至右填充的
        #                 plotter.add_volume(image_data, cmap="viridis", clim=[-1.0, 3.0])
        #
        #     # 显示所有子图
        #     plotter.link_views()  # 连接视图，使得旋转一个视图时其他视图也会跟着旋转
        #     plotter.show()

        #三维特征图切片的可视化
        # def plot_feature_map(tensor, tensor_name, vmin, vmax, channel=7):
        #     # 选择特征通道
        #     selected_channel = tensor[0, channel, :, :, :]
        #     # 选择切片，比如中间的切片
        #     slice_index = selected_channel.shape[0] // 2
        #     selected_slice = selected_channel[slice_index, :, :]
        #     # 将Tensor转换为Numpy数组进行可视化
        #     selected_slice_np = selected_slice.detach().cpu().numpy()
        #     # 可视化切片
        #     plt.imshow(selected_slice_np, cmap='viridis', vmin=vmin, vmax=vmax)  # 使用颜色映射
        #     plt.colorbar()
        #     plt.title(f'{tensor_name} Feature Map - Channel {channel}, Slice {slice_index}')
        #     plt.axis('off')
        #     plt.show()
        #
        # # 手动设置最小值和最大值
        # vmin = -1.0
        # vmax = 3.0
        # # 绘制 x3_2 的特征图
        # plot_feature_map(x3_2, 'x3_2', vmin, vmax)
        # # 绘制 layer4 的特征图
        # plot_feature_map(layer4, 'layer4', vmin, vmax)
        # # 绘制 z 的特征图
        # plot_feature_map(z, 'z', vmin, vmax)
        # # 绘制 y1 的特征图
        # plot_feature_map(y1, 'y1', vmin, vmax)
        # # 绘制 y 的特征图
        # plot_feature_map(y, 'y', vmin, vmax)
        # print(path)

        x_second = self.cbam3d_model(y, 2)
        # x_second2 = DDP_model['cbam2'](x_2_2)
        # idh_out, atrx_out, p19q_out=DDP_model['idh'](x4_1, encoder_output,x_second,x_second2)
        idh_out, atrx_out, p19q_out = self.IDH_model(x4_1, encoder_output,x_second)
        loss, idh_loss, atrx_loss, p19q_loss, idh_std, atrx_std, p19q_std, log_var_1, log_var_2, log_var_3 = self.MTL([idh_out, atrx_out, p19q_out], [idh, atrx, p19q], [weight_IDH, weight_ATRX, weight_p19q])

        # logging.info(
        #     'Epoch: {} loss: {:.5f}  idh_loss: {:.5f} atrx_loss: {:.5f} p19q_loss: {:.5f} ||idh_std:{:.4f} atrx_std:{:.4f} p19q_std:{:.4f} idh_vars:{:.4f} atrx_vars:{:.4f} p19q_vars:{:.4f}'
        #     .format(self.epoch,loss, idh_loss, atrx_loss, p19q_loss, idh_std, atrx_std, p19q_std,log_var_1, log_var_2, log_var_3))
        return loss

    def training_step(self, batch, batch_idx):

        loss = self.step(batch)
        return loss

    def on_train_epoch_end(self):
        self.epoch += 1

    def on_validation_epoch_start(self):
        self.idh_probs = []
        self.idh_class = []
        self.idh_target = []
        self.atrx_probs = []
        self.atrx_class = []
        self.atrx_target = []
        self.p19q_probs = []
        self.p19q_class = []
        self.p19q_target = []
        self.epoch_valid_loss = 0.0
        self.epoch_idh_loss = 0.0
        self.epoch_atrx_loss = 0.0
        self.epoch_p19q_loss = 0.0


    def validation_step(self, data, batch_idx):

        x, idh, atrx, p19q, x_2= data

        layer4=self.cbam3d_model(x_2,1)
        x1_1, x2_1, x3_1, x4_1, encoder_output, y,y1 = self.model(x,layer4)
        y=layer4+y
        x_second = self.cbam3d_model(y,2)

        idh_out, atrx_out, p19q_out = self.IDH_model(x4_1,encoder_output,x_second)

        valid_loss, idh_loss, atrx_loss, p19q_loss,std_1,std_2, std_3,var_1, var_2, var_3=self.MTL([idh_out, atrx_out, p19q_out], [idh, atrx, p19q],[None, None,None])
        self.epoch_valid_loss += valid_loss / 75

        self.epoch_idh_loss += idh_loss / 75
        self.epoch_atrx_loss += atrx_loss / 75
        self.epoch_p19q_loss += p19q_loss / 75

        idh_pred = F.softmax(idh_out, 1)
        # idh_pred = idh_out.sigmoid()
        idh_pred_class = torch.argmax(idh_pred, dim=1)
        # idh_pred_class = (idh_pred > 0.5).float()
        self.idh_probs.append(idh_pred[0][1].cpu())
        # idh_probs.append(idh_pred[0])
        self.idh_class.append(idh_pred_class.item())
        self.idh_target.append(idh.item())

        atrx_pred = F.softmax(atrx_out, 1)
        atrx_pred_class = torch.argmax(atrx_pred, dim=1)
        self.atrx_probs.append(atrx_pred[0][1].cpu())
        self.atrx_class.append(atrx_pred_class.item())
        self.atrx_target.append(atrx.item())

        p19q_pred = F.softmax(p19q_out, 1)
        p19q_pred_class = torch.argmax(p19q_pred, dim=1)
        self.p19q_probs.append(p19q_pred[0][1].cpu())
        self.p19q_class.append(p19q_pred_class.item())
        self.p19q_target.append(p19q.item())





    def on_validation_epoch_end(self):
        # print("test")
        accuracy_idhv = accuracy_score(self.idh_target, self.idh_class)
        auc_idhv = roc_auc_score(self.idh_target, self.idh_probs)
        accuracy_atrxv = accuracy_score(self.atrx_target, self.atrx_class)
        auc_atrxv = roc_auc_score(self.atrx_target, self.atrx_probs)
        accuracy_p19qv = accuracy_score(self.p19q_target, self.p19q_class)
        auc_p19qv = roc_auc_score(self.p19q_target, self.p19q_probs)
        # 计算混淆矩阵
        idh_tn, idh_fp, idh_fn, idh_tp = confusion_matrix(self.idh_target, self.idh_class).ravel()
        atrx_tn, atrx_fp, atrx_fn, atrx_tp = confusion_matrix(self.atrx_target, self.atrx_class).ravel()
        p19q_tn, p19q_fp, p19q_fn, p19q_tp = confusion_matrix(self.p19q_target, self.p19q_class).ravel()
        # 计算特异性和敏感度
        specificity_idhv = idh_tn / (idh_tn + idh_fp)
        sensitivity_idhv = idh_tp / (idh_tp + idh_fn)
        specificity_atrxv = atrx_tn / (atrx_tn + atrx_fp)
        sensitivity_atrxv = atrx_tp / (atrx_tp + atrx_fn)
        specificity_p19qv = p19q_tn / (p19q_tn + p19q_fp)
        sensitivity_p19qv = p19q_tp / (p19q_tp + p19q_fn)
        # print("accuracy_idhv:",accuracy_idhv)
        if accuracy_atrxv + accuracy_idhv + accuracy_p19qv > self.best_acc:
            # min_loss = epoch_valid_loss
            self.best_acc = accuracy_atrxv + accuracy_idhv + accuracy_p19qv
            self.best_epoch = self.epoch
            logging.info('there is an improvement that update the metrics and save the best model.')
            logging.info(f'Epoch {self.epoch} | '
                         f'IDH_ACC: {accuracy_idhv:.5f}, IDH_AUC: {auc_idhv:.5f}, IDH_Sensitivity: {sensitivity_idhv:.5f}, IDH_Specificity: {specificity_idhv:.5f} | '
                         f'ATRX_ACC: {accuracy_atrxv:.5f}, ATRX_AUC: {auc_atrxv:.5f}, ATRX_Sensitivity: {sensitivity_atrxv:.5f}, ATRX_Specificity: {specificity_atrxv:.5f} | '
                         f'1p19q_ACC: {accuracy_p19qv:.5f}, 1p19q_AUC: {auc_p19qv:.5f}, 1p19q_Sensitivity: {sensitivity_p19qv:.5f}, 1p19q_Specificity: {specificity_p19qv:.5f}')

            file_name = os.path.join(self.checkpoint_dir, 'model_' + str(self.epoch) + '_' + str(self.best_acc) + '_' + str(
                accuracy_idhv * 100) + '_' + str(accuracy_atrxv * 100) + '_' + str(accuracy_p19qv * 100) + '_' + str(
                auc_idhv) + '_' + str(auc_atrxv) + '_' + str(auc_p19qv) + '.pth')
            torch.save({
                'epoch': self.epoch,
                'en_state_dict': self.model.state_dict(),
                'cbam_state_dict': self.cbam3d_model.state_dict(),
                # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
                'idh_state_dict': self.IDH_model.state_dict(),
                'optim_dict': self.optimizer.state_dict(),
            },
                file_name)
        # elif self.epoch > 200 and self.epoch < 700 and self.epoch % 5 == 0:
        #     acc_1 = accuracy_atrxv + accuracy_idhv + accuracy_p19qv
        #     file_name = os.path.join(self.checkpoint_dir, 'model_' + str(self.epoch) + '_' + str(acc_1) + '_' + str(
        #         accuracy_idhv * 100) + '_' + str(accuracy_atrxv * 100) + '_' + str(accuracy_p19qv * 100) + '_' + str(
        #         auc_idhv) + '_' + str(auc_atrxv) + '_' + str(auc_p19qv) + '.pth')
        #     torch.save({
        #         'epoch': self.epoch,
        #         'en_state_dict': self.model.state_dict(),
        #         'cbam_state_dict': self.cbam3d_model.state_dict(),
        #         # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
        #         'idh_state_dict': self.IDH_model.state_dict(),
        #         'optim_dict': self.optimizer.state_dict(),
        #     },
        #         file_name)
        # elif auc_p19qv > 0.8:
        #     acc_2 = accuracy_atrxv + accuracy_idhv + accuracy_p19qv
        #     file_name = os.path.join(self.checkpoint_dir,
        #                              'auc_p19qv大于0.8_model_' + str(self.epoch) + '_' + str(acc_2) + '_' + str(
        #                                  accuracy_idhv * 100) + '_' + str(accuracy_atrxv * 100) + '_' + str(
        #                                  accuracy_p19qv * 100) + '_' + str(auc_idhv) + '_' + str(auc_atrxv) + '_' + str(
        #                                  auc_p19qv) + '.pth')
        #     torch.save({
        #         'epoch': self.epoch,
        #         'en_state_dict': self.model.state_dict(),
        #         'cbam_state_dict': self.cbam3d_model.state_dict(),
        #         # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
        #         'idh_state_dict': self.IDH_model.state_dict(),
        #         'optim_dict': self.optimizer.state_dict(),
        #     },
        #         file_name)
        logging.info(
            'Epoch:{}[best_epoch:{} ||best_acc:{:.5f}| epoch_valid_loss:{:.5f} |idh_loss: {:.5f} | atrx_loss: {:.5f} | p19q_loss: {:.5f} || idhv_acc: {:.5f} | idhv_auc:{:.5f} | idhv_sens:{:.5f} | idhv_spec:{:.5f} | atrxv_acc: {:.5f} | atrxv_auc:{:.5f} | atrxv_sens:{:.5f} | atrxv_spec:{:.5f} | p19qv_acc: {:.5f} | p19qv_auc:{:.5f} | p19qv_sens:{:.5f} | p19qv_spec:{:.5f}'
            .format(self.epoch, self.best_epoch, self.best_acc, self.epoch_valid_loss, self.epoch_idh_loss, self.epoch_atrx_loss,
                    self.epoch_p19q_loss, accuracy_idhv, auc_idhv, sensitivity_idhv, specificity_idhv,
                    accuracy_atrxv, auc_atrxv, sensitivity_atrxv, specificity_atrxv, accuracy_p19qv, auc_p19qv,
                    sensitivity_p19qv, specificity_p19qv))

    def epoch_end(self, logs, prefix):
        keys = set([key for log in logs for key in log])
        results = {key: [] for key in keys}
        for log in logs:
            for key, value in log.items():
                results[key].append(value)
        logs = {f"{prefix}/{key}": np.nanmean(results[key]) for key in keys}
        self.log_dict(logs, rank_zero_only=True)
        # if prefix == 'val':
        #     self.log('val_voxel_loss_fine', logs["val/voxel_loss_fine"], rank_zero_only=True)

    def train_dataloader(self):
        return self.dataloader("train", augment=True)

    def val_dataloader(self):
        valid_list = os.path.join(self.config.root, self.config.valid_dir, self.config.valid_file)
        valid_root = os.path.join(self.config.root, self.config.valid_dir)
        valid_set = BraTS(valid_list, valid_root, 'valid')
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        return valid_loader

    def dataloader(self, split, augment=False):

        train_list = os.path.join(self.config.root, self.config.train_dir, self.config.train_file)
        train_root = os.path.join(self.config.root, self.config.train_dir)
        train_set = BraTS(train_list, train_root, self.config.mode)

        return torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=2,
            drop_last=True,
            num_workers=self.config.num_workers
        )

    def log_args(self, log_file):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s ===> %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # args FileHandler to save log file
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # args StreamHandler to print log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # add the two Handler
        logger.addHandler(ch)
        logger.addHandler(fh)

