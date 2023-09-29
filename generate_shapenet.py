import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from dataset.HO3D_diversity_generation import HO3D_diversity
from network.affordanceNet_obman_mano_vertex import affordanceNet
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
from network.cmapnet_objhand import pointnet_reg
import numpy as np
import random
from utils import utils, utils_loss
import mano
import json
from utils.loss import TTT_loss
import trimesh
from metric.simulate import run_simulation
import pickle
from tqdm import tqdm
from pytorch3d.structures import Meshes
from utils.graspgen import GraspGenMesh,get_dgrasp_label,show_pointcloud_objhand
move_to_device = lambda dic, device: {k: v.to(device) for k, v in dic.items()}
move_to_numpy = lambda dic: {k: v.cpu().detach().numpy() for k, v in dic.items()}

def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign


def main(args, model, cmap_model, eval_loader, device, rh_mano, rh_faces):
    '''
    Generate diverse grasps for object index with args.obj_id in out-of-domain HO3D object models
    '''
    model.eval()
    cmap_model.eval()
    rh_mano.eval()


    acr_dic = 'acr_dict.pkl'
    with open(acr_dic, 'rb') as f:
        acr_dict = pickle.load(f)

    above_table = False
    optimize_gripper = True

    if __name__ == '__main__':
        # load csv file
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        parser = argparse.ArgumentParser()
        parser.add_argument('--num', '-n', default=10, type=int)
        parser.add_argument('-p', '--path', default='data', type=str)
        args = parser.parse_args()

        dataset_path = '/local/home/lafeng/Downloads/dataset'
        grasp_path = os.path.join(dataset_path, 'grasps')
        model_path = os.path.join(dataset_path, 'models')
        mp_v2_path = os.path.join(dataset_path, 'vhacd_mesh')

        grasp_list = os.listdir(grasp_path)

        gripper = trimesh.load_mesh('gripper.obj')

        total_cnt = 0

        for cnt, (obj_type, obj_path) in enumerate(tqdm(acr_dict.items())):
            for indx, path in enumerate(obj_path):
                # if indx>30:
                #     break
                try:
                    path, scale, trans = path[0], path[1], path[2]
                    obj_full_name = obj_type + '_' + path + '_' + str(scale)
                    p = os.path.join(mp_v2_path, obj_full_name, 'textured_simple.obj')
                    obj_mesh = trimesh.load(p)

                except:
                    continue

                bs = 2
                points, face_id = trimesh.sample.sample_surface(obj_mesh, 3000)
                normals = obj_mesh.face_normals[face_id]
                obj_mesh_gg = GraspGenMesh(obj_mesh.vertices, obj_mesh.faces, points, normals)
                obj_mesh_gg.to(device).repeat(bs)


                verts,_ = trimesh.sample.sample_surface(obj_mesh, 3000)
                obj_pc = torch.tensor(verts, dtype=torch.float32, device=device)
                obj_cat = torch.zeros([3000, 1], device=obj_pc.device)
                obj_cat[:] = 0.25
                obj_pc = torch.cat([obj_pc, obj_cat], dim=1)
                obj_pc = obj_pc.unsqueeze(0).transpose(1, 2)  # [1, 3, 3000]
                obj_xyz = obj_pc.permute(0,2,1)[:,:,:3].squeeze(0).cpu().numpy()  # [3000, 3]

                origin_verts = obj_mesh.vertices  # [N, 3]


                recon_param_lis = torch.zeros([bs,61], device=obj_pc.device)
                mask_lis = []

                for i in range(bs):
                    # generate random rotation
                    rot_angles = np.zeros(3) * np.pi * 2
                    theta_x, theta_y, theta_z = rot_angles[0], rot_angles[1], rot_angles[2]
                    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
                    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
                    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
                    rot = Rx @ Ry @ Rz  # [3, 3]
                    # generate random translation
                    trans = np.array([-0.0793, 0.0208, -0.6924])
                    trans = trans.reshape((3, 1))
                    R = np.hstack((rot, trans))  # [3, 4]
                    obj_xyz_transformed = np.matmul(R[:3,0:3], obj_xyz.copy().T) + R[:3,3].reshape(-1,1)  # [3, 3000]
                    obj_mesh_verts = (np.matmul(R[:3,0:3], origin_verts.copy().T) + R[:3,3].reshape(-1,1)).T  # [N, 3]
                    obj_xyz_transformed = torch.tensor(obj_xyz_transformed, dtype=torch.float32)
                    obj_pc_transformed = obj_pc.clone()
                    obj_pc_transformed[0, :3, :] = obj_xyz_transformed  # [1, 4, N]

                    obj_pc_TTT = obj_pc_transformed.detach().clone().to(device)
                    torch.save(obj_pc_TTT, 'obj_pc_TTT.pt')
                    recon_param = model.inference(obj_pc_TTT).detach()  # recon [1,61] mano params
                    recon_param = torch.autograd.Variable(recon_param, requires_grad=True)
                    optimizer = torch.optim.SGD([recon_param], lr=0.00000625, momentum=0.8)

                    for j in range(3):  # non-learning based optimization steps
                        optimizer.zero_grad()
                        # recon_param = np.load('recon_param.npy')
                        # recon_param = torch.tensor(recon_param, dtype=torch.float32, device=device)
                        recon_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                             hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
                        show_pointcloud_objhand(recon_mano.vertices[0].detach().cpu().numpy(),
                                                obj_pc_TTT[0, :3].detach().cpu().numpy().swapaxes(0, 1))


                        recon_xyz = recon_mano.vertices.to(device)  # [B,778,3], hand vertices

                        # calculate cmap from current hand
                        obj_nn_dist_affordance, _ = utils_loss.get_NN(obj_pc_TTT.permute(0, 2, 1)[:, :, :3], recon_xyz)
                        cmap_affordance = utils.get_pseudo_cmap(obj_nn_dist_affordance)  # [B,3000]

                        # predict target cmap by ContactNet
                        recon_cmap = cmap_model(obj_pc_TTT[:, :3, :], recon_xyz.permute(0, 2, 1).contiguous())  # [B,3000]
                        recon_cmap = (recon_cmap / torch.max(recon_cmap, dim=1)[0]).detach()

                        penetr_loss, consistency_loss, contact_loss = TTT_loss(recon_xyz, rh_faces,
                                                                               obj_pc_TTT[:, :3, :].permute(0,2,1).contiguous(),
                                                                               cmap_affordance, recon_cmap)
                        loss = 1 * contact_loss + 1 * consistency_loss + 7 * penetr_loss
                        loss.backward()
                        optimizer.step()

                    # evaluate grasp

                    obj_mesh = trimesh.Trimesh(vertices=obj_mesh_verts,
                                               faces=obj_mesh.faces)  # obj
                    final_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                         hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
                    final_mano_verts = final_mano.vertices.squeeze(0).detach().cpu().numpy()  # [778, 3]
                    try:
                        hand_mesh = trimesh.Trimesh(vertices=final_mano_verts, faces=rh_faces.cpu().numpy().reshape((-1, 3)))
                    except:
                        continue
                    #trimesh.Scene([obj_mesh, hand_mesh]).show()
                    # penetration volume
                    penetr_vol = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
                    # contact
                    penetration_tol = 0.005
                    result_close, result_distance, _ = trimesh.proximity.closest_point(obj_mesh, final_mano_verts)
                    sign = mesh_vert_int_exts(obj_mesh, final_mano_verts)
                    nonzero = result_distance > penetration_tol
                    exterior = [sign == -1][0] & nonzero
                    contact = ~exterior
                    sample_contact = contact.sum() > 0
                    # simulation displacement
                    vhacd_exe = "/local/home/lafeng/Desktop/v-hacd/app/TestVHACD"
                    try:
                        simu_disp = run_simulation(final_mano_verts, rh_faces.reshape((-1, 3)),
                                                  obj_mesh_verts, obj_mesh.faces,
                                                  vhacd_exe=vhacd_exe, sample_idx=i)
                    except:
                        simu_disp = 0.10

                    save_flag = (penetr_vol < 4e-6) and (simu_disp < 0.03) and sample_contact
                    mask_lis.append(save_flag)
                    recon_param_lis[i] = recon_param
                recon_param = recon_param_lis
                ##############################
                output = {}
                output['pos'] = recon_param[:, -3:]-torch.tensor(trans,device=device).unsqueeze(0).squeeze(-1)
                # aixs-angle to rot6d
                rot = recon_param[:, 10:13]
                rot_mat = axis_angle_to_matrix(rot)
                rot_6d = torch.cat([rot_mat[..., 0], rot_mat[..., 1]], axis=-1)
                output['rot'] = rot_6d
                output['pca'] = recon_param[:, 13:58]

                with torch.no_grad():
                    rh_mano_ = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                                        model_type='mano',
                                        use_pca=True,
                                        num_pca_comps=45,
                                        batch_size=bs,
                                        flat_hand_mean=True).to(device)

                final_mano = rh_mano_(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                     hand_pose=recon_param[:, 13:58], transl=output['pos'])

                hand_mesh = Meshes(verts=final_mano.vertices, faces=rh_faces)
                hand_normals = hand_mesh.verts_normals_padded()
                hand_mesh = GraspGenMesh(verts, rh_faces, verts, hand_normals)
                output['hand_mesh'] = hand_mesh
                output['all_mask'] = torch.tensor(mask_lis, dtype=torch.bool).to(device)
                # get output and init_pose
                init_pose = {}
                init_pose['rot'] = rot_6d
                init_pose['pca'] = recon_param[:, 13:58]
                offset = torch.tensor([[0.09566993, 0.00638343, 0.00618631]]).to(device)
                vec = (output['pos']+offset)*2-offset
                init_pose['pos'] = vec

                # if torch.sum(output['all_mask']) == 0:
                #     continue
                init_pose = move_to_device(init_pose, 'cpu')
                output = move_to_device(output, 'cpu')
                obj_mesh_gg.to('cpu')

                dgrasp_label = get_dgrasp_label(init_pose, output,
                                                np.array(points)[np.newaxis].repeat(bs, axis=0))
                dgrasp_label = remove_invalid(dgrasp_label, output['all_mask'])

                if optimize_gripper:
                    hand_mesh = output['hand_mesh'].remove_by_mask(output['all_mask'])
                    grasp_file = os.path.join(grasp_path, obj_type + '_' + path + '_' + str(scale) + '.h5')
                    T, success = load_grasps(grasp_file)
                    T[:, :3, 3] -= trans
                    T = T[success == 1]
                    if len(T) < 100:
                        continue
                    sampled_T, T_idx = sample_transformations(T, 100)
                    gripper_mesh = get_gripper_mesh(sampled_T, gripper)
                    collision_array = mesh_collision_check(hand_mesh, gripper_mesh)
                    if collision_array.sum() == 0:
                        continue
                    # for k in range(len(gripper_mesh)):
                    #     trimesh.Scene([hand_mesh.get_trimesh(0),gripper_mesh.get_trimesh(k),obj_mesh]).show()

                    dgrasp_label['gripper_T'] = sampled_T[np.newaxis].repeat(len(hand_mesh), axis=0)
                    dgrasp_label['collision_mask'] = collision_array

                # visualize_two_meshes(obj_mesh_gg, output['hand_mesh'], output['all_mask'], save_path=obj_full_name, show=True)

                dgrasp_label['obj_name'] = [obj_full_name for i in range(dgrasp_label['final_qpos'].shape[0])]
                dgrasp_label['scale'] = [scale for i in range(dgrasp_label['final_qpos'].shape[0])]
                all_label[total_cnt] = dgrasp_label
                total_cnt += 1

            break
            print(f'finish {total_cnt} objects out of {generate_number}')
            if total_cnt > generate_number:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''experiment setting'''
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--dataloader_workers", type=int, default=32)
    '''affordance network information'''
    parser.add_argument("--affordance_model_path", type=str, default='checkpoints/model_affordance_best_full.pth')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1024, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 256, 61])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--obj_inchannel", type=int, default=4)
    parser.add_argument("--condition_size", type=int, default=1024)
    '''cmap network information'''
    parser.add_argument("--cmap_model_path", type=str, default='checkpoints/model_cmap_best.pth')
    '''Generated graps information'''
    parser.add_argument("--obj_id", type=int, default=6)
    # You can change the two thresholds to save the graps you want
    parser.add_argument("--penetr_vol_thre", type=float, default=4e-6)  # 4cm^3
    parser.add_argument("--simu_disp_thre", type=float, default=0.03)  # 3cm
    parser.add_argument("--num_grasp", type=int, default=100)  # number of grasps you want to generate
    args = parser.parse_args()
    assert args.obj_id in [3, 4, 6, 10, 11, 19, 21, 25, 35, 37]

    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device", device)

    # network
    affordance_model = affordanceNet(obj_inchannel=args.obj_inchannel,
                                     cvae_encoder_sizes=args.encoder_layer_sizes,
                                     cvae_latent_size=args.latent_size,
                                     cvae_decoder_sizes=args.decoder_layer_sizes,
                                     cvae_condition_size=args.condition_size)  # GraspCVAE
    cmap_model = pointnet_reg(with_rgb=False)  # ContactNet

    # load pre-trained model
    checkpoint_affordance = torch.load(args.affordance_model_path, map_location=torch.device('cpu'))['network']
    affordance_model.load_state_dict(checkpoint_affordance)
    affordance_model = affordance_model.to(device)
    checkpoint_cmap = torch.load(args.cmap_model_path, map_location=torch.device('cpu'))['network']
    cmap_model.load_state_dict(checkpoint_cmap)
    cmap_model = cmap_model.to(device)

    # dataset
    dataset = HO3D_diversity()
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                            model_type='mano',
                            use_pca=True,
                            num_pca_comps=45,
                            batch_size=1,
                            flat_hand_mean=True).to(device)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)  # [1, 1538, 3], face indexes

    main(args, affordance_model, cmap_model, dataloader, device, rh_mano, rh_faces)

