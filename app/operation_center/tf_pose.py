"""Read local URDF joint transforms from TF; never invent a world/base pose."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from pose import JOINTS


def multiply(a, b):
    x, y, z, w = a
    u, v, s, t = b
    return [w*u+x*t+y*s-z*v, w*v-x*s+y*t+z*u,
            w*s+x*v-y*u+z*t, w*t-x*u-y*v-z*s]


class TfPose:
    def __init__(self):
        robot = ET.parse(Path(__file__).parent / 'assets/robot/R1.urdf').getroot()
        self.joints = {}
        for joint in robot.findall('joint'):
            name = joint.attrib['name']
            if name not in JOINTS:
                continue
            origin = joint.find('origin')
            xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
            r, p, y = [float(v)/2 for v in origin.get('rpy', '0 0 0').split()]
            rotation = multiply(multiply([0, 0, math.sin(y), math.cos(y)],
                                        [0, math.sin(p), 0, math.cos(p)]),
                                [math.sin(r), 0, 0, math.cos(r)])
            axis = [float(v) for v in joint.find('axis').get('xyz').split()]
            norm = math.sqrt(sum(v*v for v in axis))
            self.joints[joint.find('child').get('link')] = {
                'name': name, 'parent': joint.find('parent').get('link'),
                'xyz': xyz, 'rotation': rotation, 'axis': [v/norm for v in axis]}
        if len(self.joints) != 26:
            raise ValueError('R1 URDF에 26개 관절이 필요합니다.')

    def joint(self, transform):
        spec = self.joints.get(transform.child_frame_id)
        if spec is None:
            return None
        if transform.header.frame_id != spec['parent']:
            raise ValueError('R1 TF 부모 좌표계가 모델과 다릅니다.')
        t, q = transform.transform.translation, transform.transform.rotation
        xyz, rotation = [t.x, t.y, t.z], [q.x, q.y, q.z, q.w]
        if any(not math.isfinite(v) for v in xyz + rotation):
            raise ValueError('R1 TF에 유효하지 않은 좌표가 있습니다.')
        if any(abs(a-b) > 1e-6 for a, b in zip(xyz, spec['xyz'])):
            raise ValueError('R1 TF 관절 원점이 모델과 다릅니다.')
        norm = math.sqrt(sum(v*v for v in rotation))
        if abs(norm-1) > 1e-4:
            raise ValueError('R1 TF 회전값이 정규화되지 않았습니다.')
        rotation = [v/norm for v in rotation]
        inverse = [-v for v in spec['rotation'][:3]] + [spec['rotation'][3]]
        delta = multiply(inverse, rotation)
        projection = sum(v*a for v, a in zip(delta[:3], spec['axis']))
        if any(abs(v-projection*a) > 1e-5 for v, a in zip(delta[:3], spec['axis'])):
            raise ValueError('R1 TF 관절 회전축이 모델과 다릅니다.')
        angle = (2*math.atan2(projection, delta[3]) + math.pi) % (2*math.pi) - math.pi
        return spec['name'], angle, {'translation': xyz, 'rotation': rotation}
