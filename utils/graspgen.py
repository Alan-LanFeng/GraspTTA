
import torch
import numpy as np
import trimesh


# from pytorch3d.ops.knn import knn_points
# from pytorch3d.structures import Meshes
# from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
# from mayavi import mlab
import open3d
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
from scipy.spatial.transform import Rotation as R
MANO_TO_CONTACT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 6,
    9: 7,
    10:8,
    11:9,
    12:9,
    13:10,
    14:11,
    15:12,
    16:12,
    17:13,
    18:14,
    19:15,
    20:15,
}

class GraspGenMesh():
    def __init__(self,verts,faces,sampled_point,sampled_normal):
        # if verts is not a tensor

        self.verts = verts if torch.is_tensor(verts) else torch.tensor(verts,dtype=torch.float32).unsqueeze(0)

        self.faces = faces if torch.is_tensor(faces) else torch.tensor(faces,dtype=torch.int32).unsqueeze(0)

        self.sampled_point = sampled_point if torch.is_tensor(sampled_point) else torch.tensor(sampled_point,dtype=torch.float32).unsqueeze(0)

        self.sampled_normal = sampled_normal if torch.is_tensor(sampled_normal) else torch.tensor(sampled_normal,dtype=torch.float32).unsqueeze(0)

    def to(self,device):
        self.verts = self.verts.to(device)
        self.faces = self.faces.to(device)
        self.sampled_point = self.sampled_point.to(device)
        self.sampled_normal = self.sampled_normal.to(device)
        return self

    def repeat(self,n):
        self.verts = torch.repeat_interleave(self.verts,n,dim=0)
        self.faces = torch.repeat_interleave(self.faces,n,dim=0)
        self.sampled_point = torch.repeat_interleave(self.sampled_point,n,dim=0)
        self.sampled_normal = torch.repeat_interleave(self.sampled_normal,n,dim=0)
        return self

    def __len__(self):
        return self.verts.shape[0]

    def get_trimesh(self,idx):
        return trimesh.Trimesh(vertices=self.verts[idx].detach().cpu().numpy(),faces=self.faces[idx].detach().cpu().numpy())

    def remove_by_mask(self,mask):
        self.verts = self.verts[mask]
        self.faces = self.faces[mask]
        self.sampled_point = self.sampled_point[mask]
        self.sampled_normal = self.sampled_normal[mask]
        return self
def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix
def get_mano_from_output(output):
    pca = output['pca']
    pos = output['pos']
    rot = matrix_to_axis_angle(compute_rotation_matrix_from_ortho6d(output['rot']))

    n_comps = pca.shape[-1]

    hand_axis_angle = torch.einsum('bi,ij->bj', [pca, ManoLayer(mano_root='networks').th_comps[:n_comps]])

    hand_param = torch.cat([rot, hand_axis_angle, pos], dim=1)

    # to float32
    hand_param = hand_param.float()
    return hand_param

def mano_to_dgrasp(param):
    offset = np.array([[0.09566993, 0.00638343, 0.00618631]])
    bs = param.shape[0]
    axisang = param[...,3:48].reshape(bs,-1,3).copy()

    # exchange ring finger and little finger's sequence
    temp = axisang[:,6:9].copy()
    axisang[:,6:9]=axisang[:,9:12]
    axisang[:,9:12]=temp

    # change axis angle to euler angle
    joint_rot = R.from_rotvec(axisang.reshape(-1,3))
    joint_euler = joint_rot.as_euler('XYZ').reshape(bs,-1)

    global_rot = R.from_rotvec(param[:,:3])
    global_euler = global_rot.as_euler('XYZ')

    # add an offset so that the wrist is the center
    hand_pos = param[:,48:]+offset
    dgrasp_qpos = np.concatenate([hand_pos,global_euler,joint_euler],axis=-1)

    return dgrasp_qpos

def show_pointcloud_objhand(hand, obj):
    '''
    Draw hand and obj xyz at the same time
    :param hand: [778, 3]
    :param obj: [3000, 3]
    '''

    handObj = np.vstack((hand, obj))

    hand_num = hand.shape[0]
    obj_num = obj.shape[0]

    c_hand, c_obj = np.array([[1, 0, 0]]), np.array([[0, 0, 1]]) # RGB
    c_hand = np.repeat(c_hand, repeats=hand_num, axis=0) # [778,3]
    c_obj = np.repeat(c_obj, repeats=obj_num, axis=0) # [3000,3]
    c_hanObj = np.vstack((c_hand, c_obj)) # [778+3000, 3]

    pc = open3d.geometry.PointCloud()
    #pcd.points = open3d.utility.Vector3dVector(np_points)

    # improve size of the points

    pc.points = open3d.utility.Vector3dVector(handObj)
    pc.colors = open3d.utility.Vector3dVector(c_hanObj)
   # pc.scale(100, center=pc.get_center()) # increase the size of the points
    # increase the size of the points

    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin=[0,0,0])
    open3d.visualization.draw_geometries([pc, frame])
def get_dgrasp_label(init_pose, output,obj_pcd,label=None):
    bs,n_comps = output['pca'].shape
    # w = get_rough_weight(obj_pcd,1)
    # ratio = label['obj_w_stacked']/w

    if label:
        obj_pos = label['obj_pose_reset'][:,:3]
    else:
        obj_pos = np.zeros([bs,3])
        obj_pos[:,0] = 0.5
        obj_pos[:,1] = 0.25
        obj_lowest = obj_pcd[:,:,2].min(axis=1)
        obj_pos[:,2] = -obj_lowest+0.5

    init_pose['pos'] += obj_pos
    output['pos'] += obj_pos
    obj_pcd += obj_pos[:, np.newaxis, :]

    new_label = {}

    mano_layer = ManoLayer(mano_root='networks', use_pca=False, ncomps=n_comps)

    hand_param_final = get_mano_from_output(output)
    hand_param_init = get_mano_from_output(init_pose)

    verts, joints = mano_layer(th_pose_coeffs=hand_param_final[:, :-3], th_trans=hand_param_final[:, -3:])
    verts /= 1000
    joints /= 1000
    #show_pointcloud_objhand(verts[0].numpy(), obj_pcd[0])

    dgrasp_qpos = mano_to_dgrasp(hand_param_final.detach().numpy())

    full_joints = torch.zeros([bs,21,3])
    full_joints[:,0] = joints[:,0]
    full_joints[:,1:-4] = joints[:,5:]
    full_joints[:,-4:] = joints[:,1:5]

    new_label['final_qpos'] = dgrasp_qpos
    new_label['final_ee'] = full_joints.reshape(bs,-1).detach().numpy()
    new_label['final_pose'] = dgrasp_qpos[:,3:]

    contact_threshold = 0.015
    ftip_pos = new_label['final_ee'].reshape(bs,-1,3)#[:,:1]

    target_contacts = np.zeros([bs,16])
    for i in range(bs):
        obj_pcd_i = obj_pcd[i]

        final_relftip_pos = np.tile(ftip_pos[[i]],(obj_pcd_i.shape[0],1,1))
        obj_verts = obj_pcd_i[:,np.newaxis]
        diff_vert_fpos = np.linalg.norm(final_relftip_pos-obj_verts,axis=-1)

        min_vert_dist = np.min(diff_vert_fpos,axis=0)

        idx_below_thresh = np.where(min_vert_dist<contact_threshold)[0]
        target_idxs = [MANO_TO_CONTACT[idx] for idx in idx_below_thresh]

        target_contacts[i,target_idxs]=1
        target_contacts[i,-1]=1

    new_label['final_contacts'] = target_contacts

    new_label['qpos_reset'] = mano_to_dgrasp(hand_param_init.detach().numpy())

    if label:
        new_label['final_obj_pos'] = label['obj_pose_reset']
        new_label['obj_pose_reset'] = label['obj_pose_reset']
        new_label['obj_w_stacked'] = label['obj_w_stacked']
        new_label['obj_idx_stacked'] = label['obj_idx_stacked']
        new_label['obj_dim_stacked'] = label['obj_dim_stacked']
        new_label['obj_type_stacked'] = label['obj_type_stacked'].astype(np.int32)

    else:
        obj_pos_reset = np.zeros([bs,7])
        obj_pos_reset[:,3] = 1
        obj_pos_reset[:,:3] = obj_pos
        new_label['final_obj_pos'] = obj_pos_reset
        new_label['obj_pose_reset'] = obj_pos_reset
        new_label['obj_w_stacked'] = get_rough_weight(obj_pcd)
        new_label['obj_dim_stacked'] = np.zeros([bs,3])
        new_label['obj_type_stacked'] = np.full([bs],2)

    return new_label

def get_rough_weight(obj_pcd,coeff=200):
    obj_x_size = obj_pcd[:,:,0].max(axis=1) - obj_pcd[:,:,0].min(axis=1)
    obj_y_size = obj_pcd[:,:,1].max(axis=1) - obj_pcd[:,:,1].min(axis=1)
    obj_z_size = obj_pcd[:,:,2].max(axis=1) - obj_pcd[:,:,2].min(axis=1)
    obj_volume = obj_x_size * obj_y_size * obj_z_size
    obj_weight = obj_volume * coeff

    return np.array(obj_weight)