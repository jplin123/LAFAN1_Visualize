import os, argparse, numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch, time, joblib
from scipy.spatial.transform import Rotation as R
import mujoco
print(mujoco.__version__)

G1_ROTATION_AXIS = torch.tensor([[
    [0, 1, 0], # l_hip_pitch 
    [1, 0, 0], # l_hip_roll
    [0, 0, 1], # l_hip_yaw
    
    [0, 1, 0], # l_knee
    [0, 1, 0], # l_ankle_pitch
    [1, 0, 0], # l_ankle_roll
    
    [0, 1, 0], # r_hip_pitch
    [1, 0, 0], # r_hip_roll
    [0, 0, 1], # r_hip_yaw
    
    [0, 1, 0], # r_knee
    [0, 1, 0], # r_ankle_pitch
    [1, 0, 0], # r_ankle_roll
    
    [0, 0, 1], # waist_yaw_joint
    [1, 0, 0], # waist_roll_joint
    [0, 1, 0], # waist_pitch_joint
   
    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_shoulder_roll
    [0, 0, 1], # l_shoulder_yaw
    
    [0, 1, 0], # l_elbow
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_shoulder_roll
    [0, 0, 1], # r_shoulder_yaw
    
    [0, 1, 0], # r_elbow
    ]])


class MotionPlayer:
    def __init__(self, args):
        # init args
        self.args = args
        if self.args.robot_type == 'g1':
            urdf_path = "robot_description/g1/g1_29dof_rev_1_0.urdf"
            self.model = mujoco.MjModel.from_xml_path(urdf_path)
            self.data = mujoco.MjData(self.model)
            self.Tpose = np.array([0,0,0.785,0,0,0,1,
                                    -0.15,0,0,0.3,-0.15,0,
                                    -0.15,0,0,0.3,-0.15,0,
                                    0,0,0,
                                    0, 1.57,0,1.57,0,0,0,
                                    0,-1.57,0,1.57,0,0,0]).astype(np.float32)
        elif self.args.robot_type == 'h1_2':
            urdf_path = "robot_description/h1_2/h1_2_wo_hand.urdf"
        elif self.args.robot_type == 'h1':
            urdf_path = "robot_description/h1/h1.urdf"
        
        # inital gym
        self.gym = gymapi.acquire_gym()
        # create sim environment
        self.sim = self._create_simulation()
        # add plane
        self._add_ground_plane()
        # load urdf
        self.asset = self._load_urdf(urdf_path)
        # create and add robot
        self.env = self._create_env_with_robot()

    def _create_simulation(self):
        """create physics simulation environment"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 30.0
        sim_params.gravity = gymapi.Vec3(0.0, 0, -9.81)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        
        return self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    def _add_ground_plane(self):
        """add plane"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # Z-up plane
        plane_params.distance = 0                   # the distance from plane to original
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

    def _load_urdf(self, urdf_path):
        """load URDF"""
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not existent: {urdf_path}")
            
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)
        
        return self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _create_env_with_robot(self):
        """create environment with robot"""
        env = self.gym.create_env(self.sim, 
                                 gymapi.Vec3(-2, 0, -2), 
                                 gymapi.Vec3(2, 2, 2), 
                                 1)
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # put on the place with 1 meter high
        self.actor = self.gym.create_actor(env, self.asset, pose, "Robot", 0, 0)
        
        return env

    def set_camera(self, viewer):
        """ set the camera"""
        cam_pos = gymapi.Vec3(3, 2, 3)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
        
    def run_viewer(self):
        """run visualize"""
        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if not self.viewer:
            return
            
        self.set_camera(self.viewer)
        motion_data = self.load_data()

        num_dofs = self.gym.get_asset_dof_count(self.asset)
        if num_dofs == 0:
            raise RuntimeError("Asset has no DOFs — check URDF import")
        
        root_state_tensor = torch.zeros((1, 13), dtype=torch.float32)
        dof_state_tensor = torch.zeros((num_dofs, 2), dtype=torch.float32)
        
        root_trans_all, root_rot_all, dof_pos_all, rot_vec_all = [], [], [], []
        
        max_motion_length = 400  # motion_data.shape[0]
        # main loop
        while not self.gym.query_viewer_has_closed(self.viewer):
            for frame_nr in range(80, max_motion_length):
                start_time = time.time()
                configuration = torch.from_numpy(motion_data[frame_nr, :])
                
                root, rot, rv = self.get_key_point(configuration)
                
                root_trans_all.append(torch.from_numpy(root))
                rot_vec_all.append(torch.from_numpy(rv))
                root_rot_all.append(torch.from_numpy(rot))
                dof_pos_all.append(configuration[7:])

                # update Isaac viewer
                root_state_tensor[0, :7] = configuration[:7]
                dof_state_tensor[:, 0] = configuration[7:]
                
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state_tensor))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state_tensor))
                
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                elapsed_time = time.time() - start_time
                sleep_time = max(0, 1.0 / 30.0 - elapsed_time)
                time.sleep(sleep_time)

            root_trans_all = torch.cat(root_trans_all, dim=0).view(-1, 3).float()
            root_rot_all = torch.cat(root_rot_all, dim=0).view(-1, 4).float()
            dof_pos_all = torch.cat(dof_pos_all, dim=0).view(-1, 29).float()
            dof_pos_all = torch.cat((dof_pos_all[:, :19], dof_pos_all[:, 22:26]), dim=1).float()
            rot_vec_all = torch.cat(rot_vec_all, dim=0).view(-1, 3).float()
            N = rot_vec_all.shape[0]
            pose_aa = torch.cat([rot_vec_all[None, :, None], G1_ROTATION_AXIS * dof_pos_all[None,:,:,None], torch.zeros((1, N, 3, 3))], axis = 2)
            
            data_name = self.args.robot_type + '_' + self.args.file_name
            data_dump = {}
            
            data_dump[data_name]={
                "root_trans_offset": root_trans_all.cpu().detach().numpy(),
                "pose_aa": pose_aa.squeeze().cpu().detach().numpy(),   
                "dof": dof_pos_all.detach().cpu().numpy(), 
                "root_rot": root_rot_all.cpu().numpy(),
                "fps": 30
                }
            joblib.dump(data_dump, "pkl_data/" + self.args.robot_type + "/" + self.args.file_name + ".pkl")
            print("retargte data save succefully!")
            break
            
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def load_data(self):
        file_name = self.args.file_name
        robot_type = self.args.robot_type
        csv_files = robot_type + '/' + file_name + '.csv'
        data = np.genfromtxt(csv_files, delimiter=',')
        
        return data
    
    # key point visualization
    def clear_lines(self):
        self.gym.clear_lines(self.viewer)

    def draw_sphere(self, pos, radius, color, env_id, pos_id=None):
        sphere_geom_marker = gymutil.WireframeSphereGeometry(radius, 20, 20, None, color=color)
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
        gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.env, sphere_pose)

    def draw_line(self, start_point, end_point, color, env_id):
        gymutil.draw_line(start_point, end_point, color, self.gym, self.viewer, self.env)
    
    # get key point
    def get_key_point(self, cfg):
        self.clear_lines()
        # Assign root and dof positions to MuJoCo model
        self.data.qpos = cfg[7:]
        mujoco.mj_forward(self.model, self.data)

        # Extract pelvis and compute rotvec
        bid = self.model.body('torso_link').id
        pos = self.data.xpos[bid].copy()
        mat = self.data.xmat[bid].reshape(3,3).copy()
        rv = R.from_matrix(mat).as_rotvec()

        # Optionally draw others (not shown)
        return pos, mat.flatten(), rv

    def get_robot_state(self):
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self.num_envs = 1
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]
        
    def save_data(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='walk1_subject1')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    args = parser.parse_args()
    
    loader = MotionPlayer(args)
    loader.run_viewer()