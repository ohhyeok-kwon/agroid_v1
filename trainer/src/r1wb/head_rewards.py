"""Geometry-only head objectives. No target overrides, hit clocks or arm rules.

The 13x13 privileged downward ray grid is reward-only. Actor still receives the
unchanged 13x9 scan. FOV is a declared virtual camera, not a calibrated R1 lens.
"""
import math
from dataclasses import dataclass
import torch
from mjlab.utils.lab_api.math import quat_apply,quat_apply_inverse,yaw_quat
from r1wb.performance import cached_read

OPTICS=dict(vertical_fov_deg=90.,aspect=4./3.,near_m=.05,far_m=3.,lens_offsets_m=((.0118,.03,0.),(.0118,-.03,0.)),
            camera_frame='camera_link: +X forward, +Y left, +Z up',camera_pitch_rad=.34907,
            calibrated=False,visibility='both stereo frusta, terrain-ray hits; occlusion not tested')
RAY_AXIS=(-1.5,-1.2,-.9,-.6,-.3,-.15,0.,.15,.3,.6,.9,1.2,1.5)
RAY_GRID=dict(size_m=(3.,3.),axis_m=RAY_AXIS,points=169,forward_m=(.3,1.5),half_corridor_m=.2,
              measure='nonuniform fixed point-sample coverage, not exact continuous area')
WEIGHTS=dict(head_view=.10,head_camera_motion=-.05,head_effort=-.05)
DIAGNOSTIC_COLUMNS=('view_fraction','camera_rms_radps','pitch_torque_Nm','yaw_torque_Nm','motion_cost','effort_cost','valid','target_count','hit_fraction','ray_center_shift_m')


@dataclass
class GazeGridPatternCfg:
    """13x13 symmetric fixed samples, including .3..1.5 m and ±.15 m strip.

    Generates ordinary terrain-only vertical rays; choosing the command-aligned
    subset happens in reward arithmetic, never in an action or a hit schedule.
    """
    axis_m: tuple[float,...] = RAY_AXIS

    def generate_rays(self,mj_model,device):
        del mj_model
        axis=torch.tensor(self.axis_m,device=device,dtype=torch.float32)
        x,y=torch.meshgrid(axis,axis,indexing='xy')
        offsets=torch.stack((x.flatten(),y.flatten(),torch.zeros_like(x).flatten()),dim=-1)
        directions=torch.zeros_like(offsets);directions[:,2]=-1.
        return offsets,directions


def geometry_values(points_w,hit,root_pos,root_quat,camera_pos,camera_quat,camera_omega_w,torque,goal_xy):
    """Pure tensor arithmetic used by runtime and CPU tests; no simulation calls."""
    heading=yaw_quat(root_quat)
    planar=quat_apply_inverse(heading[:,None,:].expand(-1,points_w.shape[1],-1),points_w-root_pos[:,None,:])[...,:2]
    speed=goal_xy.norm(dim=-1,keepdim=True)
    direction=goal_xy/speed.clamp_min(1e-8)
    forward=torch.zeros_like(direction);forward[:,0]=1.
    direction=torch.where(speed>1e-6,direction,forward)
    along=(planar*direction[:,None,:]).sum(-1)
    across=planar[...,0]*direction[:,None,1]-planar[...,1]*direction[:,None,0]
    target=(along>=.3-1e-6)&(along<=1.5+1e-6)&(across.abs()<=.2+1e-6)
    local=quat_apply_inverse(camera_quat[:,None,:].expand(-1,points_w.shape[1],-1),points_w-camera_pos[:,None,:])
    tan_y=math.tan(math.radians(OPTICS['vertical_fov_deg']/2.));tan_x=tan_y*OPTICS['aspect']
    visible=torch.ones_like(hit,dtype=torch.bool)
    for lens in OPTICS['lens_offsets_m']:
        offset=local-local.new_tensor(lens)
        depth=offset[...,0]
        visible &= (depth>=OPTICS['near_m'])&(depth<=OPTICS['far_m'])&(offset[...,1].abs()<=depth*tan_x)&(offset[...,2].abs()<=depth*tan_y)
    count=target.sum(-1)
    view=(visible&target&hit).sum(-1)/count.clamp_min(1)
    rms=camera_omega_w.square().mean(-1).sqrt()
    motion=rms/(1.+rms) # reference angular speed 1 rad/s, bounded [0,1).
    effort=(torque.abs()/torque.new_tensor([33.,.86])).clamp(max=1.).mean(-1)
    # No hits or no points in the corridor means measured coverage zero, not a
    # nonfinite physical state. Keep their counts separately; never exempt time.
    valid=torch.isfinite(points_w).all(dim=(-1,-2))&torch.isfinite(camera_pos).all(-1)&torch.isfinite(camera_quat).all(-1)&torch.isfinite(camera_omega_w).all(-1)&torch.isfinite(torque).all(-1)&torch.isfinite(root_pos).all(-1)&torch.isfinite(root_quat).all(-1)&torch.isfinite(goal_xy).all(-1)
    result=torch.stack((view,rms,torque[:,0],torque[:,1],motion,effort,valid.float(),count.float(),(target&hit).sum(-1)/count.clamp_min(1)),dim=1)
    return result


@cached_read
def diagnostic_values(env):
    entity=env.scene['robot'];data=entity.data
    if '_r1wb_head_ids' not in env.__dict__:
        env._r1wb_head_ids=(entity.body_names.index('camera_link'),[entity.joint_names.index(n) for n in ('head_pitch_joint','head_yaw_joint')])
    body,joints=env._r1wb_head_ids
    ray=env.scene['head_gaze_ground'].data
    goal=env.command_manager.get_term('motion').goal[:,:2]
    result=geometry_values(ray.hit_pos_w,ray.distances>=0,data.root_link_pos_w,data.root_link_quat_w,
        data.body_link_pos_w[:,body],data.body_link_quat_w[:,body],data.body_link_ang_vel_w[:,body],data.qfrc_actuator[:,joints],goal)
    # Native mjlab senses after rewards, once per 20 ms control step. On static
    # terrain the saved world hit points remain valid surface points. Test them
    # against the current camera, and record how far their sampling centre has
    # moved. Do not call sense/forward here or change legacy body reward timing.
    # A moving/deforming terrain requires a different, freshly validated sampler.
    displacement=(ray.frame_pos_w[:,0]-data.root_link_pos_w).norm(dim=-1)
    result=torch.cat((result,displacement[:,None]),dim=1)
    result[:,6]*=(torch.isfinite(ray.distances).all(-1)&torch.isfinite(displacement)).float()
    good=result[:,6]>0
    # The zero-weight h0 arm is also audited by the pure vector callback.
    # Actual nonfinite data remains invalid in both arms; normal sensor lag does
    # not create a nonphysical NaN. Columns 7/8 expose missing target/hit samples.
    for column in (0,1,2,3,4,5,8):
        result[:,column]=torch.where(good,result[:,column],torch.full_like(result[:,column],float('nan')))
    return result


def _cost(env,column):
    values=diagnostic_values(env)
    # Raw nonfinite values are captured by the strict reward ledger; never hide a
    # failed geometry/sensor contract behind a finite arbitrary reward.
    return torch.where(values[:,6]>0,values[:,column],torch.full_like(values[:,column],float('nan')))


def view_fraction(env):return _cost(env,0)
def camera_motion_cost(env):return _cost(env,4)
def head_effort_cost(env):return _cost(env,5)
