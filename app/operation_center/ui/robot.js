import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import startup from '../startup.json';
import sensors from '../assets/robot/sensors.json';

function vector(value, fallback = [0, 0, 0]) {
  return value ? value.trim().split(/\s+/).map(Number) : fallback;
}

function origin(object, node) {
  object.position.fromArray(vector(node?.getAttribute('xyz')));
  const rpy = vector(node?.getAttribute('rpy'));
  object.quaternion.setFromEuler(new THREE.Euler(...rpy, 'ZYX'));
}

export class RobotView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x11151b);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.4;
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x11151b, 5, 14);
    this.camera = new THREE.PerspectiveCamera(startup.view.fov, 1, .02, 60);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = .09;
    this.controls.minDistance = .75;
    this.controls.maxDistance = 8;
    this.controls.maxPolarAngle = Math.PI * .51;
    this.controls.target.set(0, 0, .64);
    this.preset('perspective');
    this.scene.add(new THREE.HemisphereLight(0xc6dcff, 0x536174, 2.3));
    const key = new THREE.DirectionalLight(0xfff3e4, 4.1);
    key.position.set(2, -3, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: .1, far: 10 });
    key.shadow.normalBias = .018;
    key.shadow.bias = -.0001;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xa5c7ff, 2);
    fill.position.set(-3, 2, 3);
    this.scene.add(fill);
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x151c26, roughness: .96, metalness: .06 }));
    this.floor.receiveShadow = true;
    this.floor.position.z = -.007;
    this.scene.add(this.floor);
    this.grid = new THREE.GridHelper(12, 60, 0x465b75, 0x283647);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.material.transparent = true;
    this.grid.material.opacity = .32;
    this.scene.add(this.grid);
    this.model = new THREE.Group();
    this.scene.add(this.model);
    this.joints = new Map();
    this.jointMarkers = new THREE.Group();
    this.scene.add(this.jointMarkers);
    this.jointMarkers.visible = false;
    this.markerPairs = [];
    this.sensorPairs = [];
    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'sensor-overlay';
    this.labels.domElement.hidden = true;
    canvas.parentElement.append(this.labels.domElement);
    this.sensorNote = document.createElement('div');
    this.sensorNote.className = 'sensor-note';
    this.sensorNote.textContent = sensors.unlocated.join(' · ');
    this.sensorNote.hidden = true;
    canvas.parentElement.append(this.sensorNote);
    this.bounds = new THREE.Box3();
    this.updating = false;
    this.resize = new ResizeObserver(() => {
      const { width, height } = canvas.getBoundingClientRect();
      if (!width || !height) return;
      this.renderer.setSize(width, height, false);
      this.labels.setSize(width, height);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.dirty = true;
    });
    this.resize.observe(canvas);
    this.frame = () => {
      const changed = this.controls.update();
      if (changed || this.dirty) {
        this.renderer.render(this.scene, this.camera);
        this.labels.render(this.scene, this.camera);
        this.dirty = false;
      }
      this.animation = requestAnimationFrame(this.frame);
    };
    this.frame();
    canvas.addEventListener('dblclick', () => { if (!this.rightDragOnly) this.preset('perspective'); });
    this.contextLost = (event) => { event.preventDefault(); cancelAnimationFrame(this.animation); };
    canvas.addEventListener('webglcontextlost', this.contextLost);
    canvas.addEventListener('webglcontextrestored', () => { this.dirty = true; this.frame(); });
  }

  preset(name) {
    const positions = { front: [3.4, 0, 1.05], side: [0, -3.4, 1.05] };
    this.camera.position.fromArray(positions[name] || startup.view.position);
    this.controls.target.fromArray(positions[name] ? [0, 0, .66] : startup.view.target);
    this.controls.update();
    this.dirty = true;
  }

  rightDragLook() {
    if (this.rightDragOnly) return;
    this.rightDragOnly = true;
    this.controls.mouseButtons = { LEFT: null, MIDDLE: null, RIGHT: THREE.MOUSE.ROTATE };
    this.controls.enableDamping = false;
    this.controls.enablePan = false;
    this.canvas.classList.add('right-drag-look');
    this.rightDragStart = (event) => { if (event.button === 2) this.canvas.classList.add('looking'); };
    this.rightDragEnd = () => this.canvas.classList.remove('looking');
    this.canvas.addEventListener('pointerdown', this.rightDragStart);
    window.addEventListener('pointerup', this.rightDragEnd);
  }

  terrain(objects) {
    if (this.obstacles) {
      this.obstacles.traverse((item) => { item.geometry?.dispose(); item.material?.dispose(); });
      this.scene.remove(this.obstacles);
    }
    this.obstacles = new THREE.Group();
    for (const item of objects) {
      const geometry = item.shape === 'box' ? new THREE.BoxGeometry(...item.size.map((v) => v * 2)) : new THREE.SphereGeometry(1, 18, 12);
      const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: item.color, roughness: .95 }));
      if (item.shape !== 'box') mesh.scale.fromArray(item.size);
      mesh.position.fromArray(item.position); mesh.rotation.z = item.yaw;
      mesh.castShadow = mesh.receiveShadow = true; this.obstacles.add(mesh);
    }
    this.scene.add(this.obstacles); this.dirty = true;
  }

  async load() {
    const response = await fetch('/assets/robot/R1.urdf');
    if (!response.ok) throw new Error('로봇 모델 파일을 읽지 못했습니다.');
    const xml = new DOMParser().parseFromString(await response.text(), 'application/xml');
    if (xml.querySelector('parsererror')) throw new Error('로봇 모델 형식이 올바르지 않습니다.');
    const robot = xml.documentElement;
    const materials = new Map();
    for (const node of robot.querySelectorAll(':scope > material')) {
      materials.set(node.getAttribute('name'), vector(node.querySelector('color')?.getAttribute('rgba'), [.72, .72, .72, 1]));
    }
    const links = new Map();
    const loads = [];
    const loader = new STLLoader();
    const geometryCache = new Map();
    for (const link of robot.querySelectorAll(':scope > link')) {
      const group = new THREE.Group();
      group.name = link.getAttribute('name');
      links.set(group.name, group);
      for (const visual of link.querySelectorAll(':scope > visual')) {
        const geometry = visual.querySelector('geometry > mesh');
        if (!geometry) continue;
        const filename = geometry.getAttribute('filename');
        if (!geometryCache.has(filename)) geometryCache.set(filename, loader.loadAsync(`/assets/robot/${filename}`));
        loads.push(geometryCache.get(filename).then((shape) => {
          shape.computeVertexNormals();
          shape.computeBoundingBox();
          const matNode = visual.querySelector('material');
          const rgba = vector(matNode?.querySelector('color')?.getAttribute('rgba'), materials.get(matNode?.getAttribute('name')) || [.72, .72, .72, 1]);
          const brightness = (rgba[0] + rgba[1] + rgba[2]) / 3;
          const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(...rgba.slice(0, 3)), roughness: brightness > .5 ? .44 : .7, metalness: .12 });
          const mesh = new THREE.Mesh(shape, material);
          mesh.name = group.name;
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          mesh.scale.fromArray(vector(geometry.getAttribute('scale'), [1, 1, 1]));
          origin(mesh, visual.querySelector('origin'));
          group.add(mesh);
        }));
      }
    }
    const children = new Set();
    for (const node of robot.querySelectorAll(':scope > joint')) {
      const parent = links.get(node.querySelector('parent').getAttribute('link'));
      const child = links.get(node.querySelector('child').getAttribute('link'));
      if (!parent || !child) throw new Error('로봇 관절 연결이 올바르지 않습니다.');
      const pivot = new THREE.Group();
      origin(pivot, node.querySelector('origin'));
      parent.add(pivot);
      pivot.add(child);
      children.add(child.name);
      if (node.getAttribute('type') !== 'fixed') {
        const name = node.getAttribute('name');
        this.joints.set(name, { pivot, child: child.name, origin: pivot.quaternion.clone(), xyz: pivot.position.clone(), axis: new THREE.Vector3().fromArray(vector(node.querySelector('axis')?.getAttribute('xyz'), [1, 0, 0])).normalize() });
        const marker = new THREE.Mesh(new THREE.SphereGeometry(.009, 12, 8), new THREE.MeshBasicMaterial({ color: 0x77baff, depthTest: false }));
        marker.renderOrder = 5;
        this.jointMarkers.add(marker);
        this.markerPairs.push([pivot, marker]);
      }
    }
    for (const [name, link] of links) if (!children.has(name)) this.model.add(link);
    await Promise.all(loads);
    for (const sensor of sensors.markers) {
      const link = links.get(sensor.link);
      if (!link) continue;
      const marker = new THREE.Mesh(new THREE.SphereGeometry(.012, 16, 12), new THREE.MeshBasicMaterial({ color: sensor.kind === 'imu' ? 0x65deeb : 0xffc77b, depthTest: false }));
      marker.name = sensor.id;
      marker.renderOrder = 6;
      const label = document.createElement('div');
      label.className = `sensor-label ${sensor.kind}`;
      label.textContent = sensor.label;
      label.title = '기체 모델에 정의된 센서 위치';
      const tag = new CSS2DObject(label);
      tag.center.set(sensor.side === 'left' ? 1.3 : sensor.side === 'right' ? -.3 : .5, 1.5);
      marker.add(tag);
      this.jointMarkers.add(marker);
      this.sensorPairs.push({ link, marker, offset: new THREE.Vector3().fromArray(sensor.position) });
    }
    this.setPose(null);
    this.loaded = true;
    return [...this.joints.keys()];
  }

  setPose(pose, { follow = false } = {}) {
    const angle = new THREE.Quaternion();
    for (const [name, joint] of this.joints) {
      const transform = pose?.transforms?.[joint.child];
      if (transform) {
        joint.pivot.position.fromArray(transform.translation);
        joint.pivot.quaternion.fromArray(transform.rotation);
      } else if (!pose || name in pose.joints) {
        joint.pivot.position.copy(joint.xyz);
        angle.setFromAxisAngle(joint.axis, pose?.joints[name] ?? 0);
        joint.pivot.quaternion.copy(joint.origin).multiply(angle);
      } // A joint missing from this measurement keeps its last shown angle.
    }
    const q = pose?.quaternion;
    if (q) this.model.quaternion.set(q[1], q[2], q[3], q[0]);
    else this.model.quaternion.identity();
    if (pose?.position) {
      this.model.position.fromArray(pose.position);
      if (follow) {
        const target = this.model.position.clone();
        const shift = target.clone().sub(this.controls.target);
        this.camera.position.add(shift);
        this.controls.target.copy(target);
      }
    } else {
      // Live TF has no world position; anchor that display to the floor.
      this.model.position.set(0, 0, 0);
      this.model.updateMatrixWorld(true);
      this.bounds.setFromObject(this.model);
      this.model.position.z = -this.bounds.min.z;
    }
    this.model.updateMatrixWorld(true);
    for (const [pivot, marker] of this.markerPairs) pivot.getWorldPosition(marker.position);
    for (const { link, marker, offset } of this.sensorPairs) {
      marker.position.copy(offset);
      link.localToWorld(marker.position);
    }
    this.tick = pose?.tick;
    this.dirty = true;
  }

  setMarkersVisible(visible) {
    this.jointMarkers.visible = visible;
    this.labels.domElement.hidden = !visible;
    this.sensorNote.hidden = !visible || !sensors.unlocated.length;
    this.dirty = true;
  }

  destroy() {
    cancelAnimationFrame(this.animation);
    this.resize.disconnect();
    this.controls.dispose();
    this.canvas.removeEventListener('pointerdown', this.rightDragStart);
    window.removeEventListener('pointerup', this.rightDragEnd);
    this.labels.domElement.remove();
    this.sensorNote.remove();
    this.scene.traverse((object) => {
      object.geometry?.dispose();
      if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
      else object.material?.dispose();
    });
    this.renderer.dispose();
  }
}
