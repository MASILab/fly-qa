// A stylized (low-poly, not photorealistic) 3D scene: the fly, at its desk,
// reviewing the same image the 2D panel shows -- purely decorative/thematic,
// no data lives here that isn't already shown elsewhere on the page. The
// monitor's screen texture updates from the same /api/image URL used by the
// 2D thumbnail, in step with the live websocket feed.

class FlyDesk3D {
  constructor(canvasEl) {
    this.canvas = canvasEl;
    this.clock = new THREE.Clock();
    this.textureLoader = new THREE.TextureLoader();
    this.currentScreenTexture = null;

    this._initScene();
    this._buildRoom();
    this._buildDesk();
    this._buildMonitor();
    this._buildChair();
    this._buildFly();
    this._initControls();

    window.addEventListener("resize", () => this._resize());
    this._resize();
    this._animate();
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05080a);
    this.scene.fog = new THREE.Fog(0x05080a, 8, 20);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    this.camera.position.set(3.2, 2.6, 4.2);
    this.camera.lookAt(0, 1, 0);

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;

    const ambient = new THREE.AmbientLight(0x8899aa, 0.55);
    this.scene.add(ambient);

    const key = new THREE.PointLight(0xfff2cc, 1.1, 12);
    key.position.set(1.5, 3.2, 1.8);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    this.scene.add(key);

    const rim = new THREE.PointLight(0x39ff6a, 0.6, 10);
    rim.position.set(-2, 1.6, -1.5);
    this.scene.add(rim);
  }

  _buildRoom() {
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x0b1410, roughness: 0.95 });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
  }

  _buildDesk() {
    const woodMat = new THREE.MeshStandardMaterial({ color: 0x2a1f18, roughness: 0.7 });
    const desk = new THREE.Group();

    const top = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.08, 1.3), woodMat);
    top.position.y = 1.0;
    top.castShadow = true;
    top.receiveShadow = true;
    desk.add(top);

    const legGeo = new THREE.BoxGeometry(0.08, 1.0, 0.08);
    for (const [x, z] of [
      [1.2, 0.55], [-1.2, 0.55], [1.2, -0.55], [-1.2, -0.55],
    ]) {
      const leg = new THREE.Mesh(legGeo, woodMat);
      leg.position.set(x, 0.5, z);
      leg.castShadow = true;
      desk.add(leg);
    }

    this.scene.add(desk);
  }

  _buildMonitor() {
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x111716, roughness: 0.5 });
    const monitor = new THREE.Group();

    const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.09, 0.28, 12), frameMat);
    stand.position.set(0, 1.18, -0.15);
    stand.castShadow = true;
    monitor.add(stand);

    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.02, 20), frameMat);
    base.position.set(0, 1.045, -0.15);
    monitor.add(base);

    const frame = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.72, 0.05), frameMat);
    frame.position.set(0, 1.68, -0.15);
    frame.castShadow = true;
    monitor.add(frame);

    // The live screen. Starts as a dim placeholder; textured from /api/image once
    // the first classification event arrives (see setScreenImage()).
    const screenMat = new THREE.MeshBasicMaterial({ color: 0x0d1a12 });
    const screen = new THREE.Mesh(new THREE.PlaneGeometry(1.04, 0.62), screenMat);
    screen.position.set(0, 1.68, -0.122);
    monitor.add(screen);
    this.screenMesh = screen;

    // Faint glow so the "monitor" reads as a light source even before an image loads.
    const glow = new THREE.PointLight(0x39ff6a, 0.4, 3);
    glow.position.set(0, 1.68, 0.3);
    monitor.add(glow);

    this.scene.add(monitor);
  }

  _buildChair() {
    const chairMat = new THREE.MeshStandardMaterial({ color: 0x1a2620, roughness: 0.8 });
    const chair = new THREE.Group();

    const seat = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.06, 0.6), chairMat);
    seat.position.set(0, 0.55, 0.85);
    seat.castShadow = true;
    chair.add(seat);

    const back = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.6, 0.06), chairMat);
    back.position.set(0, 0.85, 1.13);
    back.castShadow = true;
    chair.add(back);

    const legGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.55, 8);
    for (const [x, z] of [[0.25, 0.6], [-0.25, 0.6], [0.25, 1.1], [-0.25, 1.1]]) {
      const leg = new THREE.Mesh(legGeo, chairMat);
      leg.position.set(x, 0.28, z);
      chair.add(leg);
    }

    this.scene.add(chair);
  }

  // A stylized (not anatomically literal) fly, low-poly, perched on the chair
  // facing the monitor. this.fly holds references used by the idle animation.
  _buildFly() {
    const fly = new THREE.Group();
    fly.position.set(0, 0.92, 0.75);
    fly.rotation.y = Math.PI; // face the monitor

    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x14201a, roughness: 0.4, metalness: 0.2 });
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0xff4d4d, emissive: 0x5a0000, roughness: 0.3 });
    const wingMat = new THREE.MeshStandardMaterial({
      color: 0x9fe8c8, transparent: true, opacity: 0.35, side: THREE.DoubleSide, roughness: 0.2,
    });
    const legMat = new THREE.MeshStandardMaterial({ color: 0x0c1512, roughness: 0.6 });
    const glowMat = new THREE.MeshStandardMaterial({
      color: 0x14201a, emissive: 0x39ff6a, emissiveIntensity: 0.15, roughness: 0.4,
    });

    const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 12), glowMat);
    abdomen.scale.set(1, 0.85, 1.5);
    abdomen.position.set(0, 0.18, -0.12);
    abdomen.castShadow = true;
    fly.add(abdomen);

    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 12), bodyMat);
    thorax.position.set(0, 0.2, 0.08);
    thorax.castShadow = true;
    fly.add(thorax);

    const head = new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 12), bodyMat);
    head.position.set(0, 0.22, 0.22);
    head.castShadow = true;
    fly.add(head);

    const eyeGeo = new THREE.SphereGeometry(0.045, 12, 10);
    const eyeL = new THREE.Mesh(eyeGeo, eyeMat);
    eyeL.position.set(0.06, 0.23, 0.27);
    fly.add(eyeL);
    const eyeR = new THREE.Mesh(eyeGeo, eyeMat);
    eyeR.position.set(-0.06, 0.23, 0.27);
    fly.add(eyeR);

    const antennaGeo = new THREE.CylinderGeometry(0.006, 0.006, 0.14, 6);
    const antennaL = new THREE.Mesh(antennaGeo, legMat);
    antennaL.position.set(0.04, 0.32, 0.24);
    antennaL.rotation.z = 0.4;
    antennaL.rotation.x = -0.3;
    fly.add(antennaL);
    const antennaR = new THREE.Mesh(antennaGeo, legMat);
    antennaR.position.set(-0.04, 0.32, 0.24);
    antennaR.rotation.z = -0.4;
    antennaR.rotation.x = -0.3;
    fly.add(antennaR);

    const wingGeo = new THREE.PlaneGeometry(0.22, 0.12);
    const wingL = new THREE.Mesh(wingGeo, wingMat);
    wingL.position.set(0.1, 0.28, 0.05);
    wingL.rotation.y = 0.5;
    wingL.rotation.z = 0.15;
    fly.add(wingL);
    const wingR = new THREE.Mesh(wingGeo, wingMat);
    wingR.position.set(-0.1, 0.28, 0.05);
    wingR.rotation.y = -0.5;
    wingR.rotation.z = -0.15;
    fly.add(wingR);

    const legGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.28, 6);
    const legs = [];
    const legAngles = [-0.9, 0, 0.9];
    for (const side of [1, -1]) {
      for (const angle of legAngles) {
        const leg = new THREE.Mesh(legGeo, legMat);
        leg.position.set(side * 0.13, 0.08, 0.06 + angle * 0.05);
        leg.rotation.z = side * 0.9;
        leg.rotation.x = angle * 0.25;
        fly.add(leg);
        legs.push(leg);
      }
    }

    this.fly = fly;
    this.flyParts = { wingL, wingR, antennaL, antennaR };
    this.scene.add(fly);
  }

  _initControls() {
    if (typeof THREE.OrbitControls !== "function") return;
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 1.1, 0.2);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 8;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.update();
  }

  // Called by app.js on each "result" websocket event with the same /api/image
  // URL the 2D thumbnail uses -- keeps the monitor showing whatever's currently
  // being classified, no separate data source.
  setScreenImage(url) {
    this.textureLoader.load(url, (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace || THREE.sRGBEncoding;
      const oldTexture = this.currentScreenTexture;
      this.screenMesh.material.map = texture;
      this.screenMesh.material.color.set(0xffffff);
      this.screenMesh.material.needsUpdate = true;
      this.currentScreenTexture = texture;
      if (oldTexture) oldTexture.dispose();
    });
  }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    this.camera.aspect = rect.width / rect.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(rect.width, rect.height, false);
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    const t = this.clock.getElapsedTime();

    if (this.flyParts) {
      const flap = Math.sin(t * 26) * 0.5;
      this.flyParts.wingL.rotation.z = 0.15 + flap;
      this.flyParts.wingR.rotation.z = -0.15 - flap;
      const antennaWiggle = Math.sin(t * 3) * 0.08;
      this.flyParts.antennaL.rotation.x = -0.3 + antennaWiggle;
      this.flyParts.antennaR.rotation.x = -0.3 - antennaWiggle;
    }
    if (this.fly) {
      this.fly.position.y = 0.92 + Math.sin(t * 1.6) * 0.01;
    }

    if (this.controls) this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
