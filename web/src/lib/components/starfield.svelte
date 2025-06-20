<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as THREE from "three";

  const CONFIG = {
    STAR_COUNT: 2000,
    FIELD_SIZE: 1000,
    CAMERA_ROTATION_SPEED: 0.001,
    CAMERA_FORWARD_SPEED: 0.1,
    COMET_SPAWN_INTERVAL_MIN: 3,
    COMET_SPAWN_INTERVAL_MAX: 8,
    MAX_COMETS_AT_ONCE: 3,
    STAR_COLORS: [
      [0.96, 0.94, 0.88], // Warm white
      [1.0, 0.96, 0.84], // Pale yellow
      [1.0, 0.92, 0.76], // Light orange
      [1.0, 0.84, 0.69], // Orange
      [0.67, 0.8, 1.0], // Blue-white
      [1.0, 0.71, 0.59], // Red-orange
    ],
  } as const;

  type Star = {
    position: THREE.Vector3;
    originalPosition: THREE.Vector3;
    color: THREE.Color;
    size: number;
    brightness: number;
    flickerSpeed: number;
    noiseOffset: number;
  };

  type Comet = {
    id: number;
    position: THREE.Vector3;
    velocity: THREE.Vector3;
    trail: THREE.Vector3[];
    life: number;
    maxLife: number;
    brightness: number;
    color: THREE.Color;
  };

  class StarField3D {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private renderer: THREE.WebGLRenderer;
    private stars: Star[] = [];
    private comets: Comet[] = [];
    private starMesh!: THREE.Points;
    private time: number = 0;
    private lastCometTime: number = 0;
    private animationId: number | null = null;

    constructor(container: HTMLElement) {
      // Scene setup
      this.scene = new THREE.Scene();
      this.scene.fog = new THREE.Fog(0x000011, 500, 1500);

      // Camera setup
      this.camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        2000,
      );
      this.camera.position.set(0, 0, 0);

      // Renderer setup
      this.renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: false, // Disable for performance
      });
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.renderer.setClearColor(0x000000, 0);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      container.appendChild(this.renderer.domElement);

      this.generateStars();
      this.setupEventListeners();
      this.animate();
    }

    private generateStars(): void {
      const positions = new Float32Array(CONFIG.STAR_COUNT * 3);
      const colors = new Float32Array(CONFIG.STAR_COUNT * 3);
      const sizes = new Float32Array(CONFIG.STAR_COUNT);

      for (let i = 0; i < CONFIG.STAR_COUNT; i++) {
        // Generate random position in sphere
        const phi = Math.random() * Math.PI * 2;
        const costheta = Math.random() * 2 - 1;
        const theta = Math.acos(costheta);
        const radius = CONFIG.FIELD_SIZE * (0.3 + Math.random() * 0.7); // Vary distance

        const x = radius * Math.sin(theta) * Math.cos(phi);
        const y = radius * Math.sin(theta) * Math.sin(phi);
        const z = radius * Math.cos(theta);

        const position = new THREE.Vector3(x, y, z);
        const originalPosition = position.clone();

        // Star properties
        const colorIndex = Math.floor(
          Math.random() * CONFIG.STAR_COLORS.length,
        );
        const [r, g, b] = CONFIG.STAR_COLORS[colorIndex];
        const color = new THREE.Color(r, g, b);

        const distance = position.length();
        const size = (1.0 + Math.random() * 3.0) * (1000 / distance); // Size based on distance
        const brightness = 0.3 + Math.random() * 0.7;

        // Store star data
        this.stars.push({
          position,
          originalPosition,
          color,
          size,
          brightness,
          flickerSpeed: 1.0 + Math.random() * 2.0,
          noiseOffset: Math.random() * 1000,
        });

        // Set geometry arrays
        const i3 = i * 3;
        positions[i3] = x;
        positions[i3 + 1] = y;
        positions[i3 + 2] = z;

        colors[i3] = r;
        colors[i3 + 1] = g;
        colors[i3 + 2] = b;

        sizes[i] = size;
      }

      // Create star geometry and material
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3),
      );
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

      const material = new THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0.0 },
        },
        vertexShader: `
          attribute float size;
          varying vec3 vColor;
          uniform float time;
          
          // Simple noise function
          float noise(float x) {
            return fract(sin(x) * 43758.5453);
          }
          
          void main() {
            vColor = color;
            
            // Twinkling effect
            float flicker = noise(time * 2.0 + position.x * 0.01) * 0.5 + 0.5;
            float finalSize = size * (0.5 + flicker * 0.5);
            
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = finalSize * (300.0 / -mvPosition.z);
            gl_Position = projectionMatrix * mvPosition;
          }
        `,
        fragmentShader: `
          varying vec3 vColor;
          
          void main() {
            float distance = length(gl_PointCoord - vec2(0.5));
            float alpha = 1.0 - smoothstep(0.0, 0.5, distance);
            
            gl_FragColor = vec4(vColor, alpha);
          }
        `,
        transparent: true,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
      });

      this.starMesh = new THREE.Points(geometry, material);
      this.scene.add(this.starMesh);
    }

    private updateCamera(): void {
      // Slow rotation around Y axis
      const rotationRadius = 50;
      const rotationSpeed = this.time * CONFIG.CAMERA_ROTATION_SPEED;

      this.camera.position.x = Math.cos(rotationSpeed) * rotationRadius;
      this.camera.position.z = Math.sin(rotationSpeed) * rotationRadius;

      // Gentle forward movement
      this.camera.position.y += CONFIG.CAMERA_FORWARD_SPEED * 0.1;

      // Look slightly ahead in the direction of movement
      const lookTarget = new THREE.Vector3(
        this.camera.position.x * 0.1,
        this.camera.position.y + 100,
        this.camera.position.z * 0.1,
      );
      this.camera.lookAt(lookTarget);
    }

    private spawnCometBurst(): void {
      const numComets =
        Math.floor(Math.random() * CONFIG.MAX_COMETS_AT_ONCE) + 1;

      for (let i = 0; i < numComets; i++) {
        this.spawnComet();
      }
    }

    private spawnComet(): void {
      // Random start position around the edge of view
      const angle = Math.random() * Math.PI * 2;
      const distance = CONFIG.FIELD_SIZE * 0.8;

      const startPos = new THREE.Vector3(
        Math.cos(angle) * distance + (Math.random() - 0.5) * 200,
        (Math.random() - 0.5) * 400,
        Math.sin(angle) * distance + (Math.random() - 0.5) * 200,
      );

      // Random velocity toward center with some variation
      const velocity = new THREE.Vector3(
        -startPos.x * 0.001 + (Math.random() - 0.5) * 0.5,
        (Math.random() - 0.5) * 0.3,
        -startPos.z * 0.001 + (Math.random() - 0.5) * 0.5,
      );

      const colorIndex = Math.floor(Math.random() * CONFIG.STAR_COLORS.length);
      const [r, g, b] = CONFIG.STAR_COLORS[colorIndex];

      this.comets.push({
        id: this.comets.length,
        position: startPos,
        velocity,
        trail: [],
        life: 1.0,
        maxLife: 1.0,
        brightness: 0.8 + Math.random() * 0.2,
        color: new THREE.Color(r, g, b),
      });
    }

    private updateComets(deltaTime: number): void {
      for (let i = this.comets.length - 1; i >= 0; i--) {
        const comet = this.comets[i];

        // Update position
        comet.position.add(
          comet.velocity.clone().multiplyScalar(deltaTime * 60),
        );

        // Add to trail
        comet.trail.push(comet.position.clone());
        if (comet.trail.length > 20) {
          comet.trail.shift();
        }

        // Update life
        comet.life -= deltaTime * 0.3;

        // Remove if dead or too far
        if (
          comet.life <= 0 ||
          comet.position.length() > CONFIG.FIELD_SIZE * 1.5
        ) {
          this.comets.splice(i, 1);
        }
      }
    }

    private renderComets(): void {
      // Remove old comet objects
      const cometsToRemove = this.scene.children.filter(
        (child) => child.userData.type === "comet",
      );
      cometsToRemove.forEach((comet) => this.scene.remove(comet));

      // Render current comets
      this.comets.forEach((comet) => {
        if (comet.trail.length < 2) return;

        // Create comet trail geometry
        const points = comet.trail.map((pos) => pos.clone());
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        const material = new THREE.LineBasicMaterial({
          color: comet.color,
          transparent: true,
          opacity: comet.life * comet.brightness,
          blending: THREE.AdditiveBlending,
        });

        const line = new THREE.Line(geometry, material);
        line.userData.type = "comet";
        this.scene.add(line);

        // Add comet head
        const headGeometry = new THREE.SphereGeometry(2, 8, 8);
        const headMaterial = new THREE.MeshBasicMaterial({
          color: comet.color,
          transparent: true,
          opacity: comet.life * comet.brightness,
          blending: THREE.AdditiveBlending,
        });

        const head = new THREE.Mesh(headGeometry, headMaterial);
        head.position.copy(comet.position);
        head.userData.type = "comet";
        this.scene.add(head);
      });
    }

    private animate = (): void => {
      this.animationId = requestAnimationFrame(this.animate);

      const currentTime = performance.now() * 0.001;
      const deltaTime = currentTime - this.time;
      this.time = currentTime;

      // Update camera movement
      this.updateCamera();

      // Update star twinkling
      if (this.starMesh.material instanceof THREE.ShaderMaterial) {
        this.starMesh.material.uniforms.time.value = this.time;
      }

      // Update comets
      this.updateComets(deltaTime);
      this.renderComets();

      // Spawn new comets
      const cometSpawnTime =
        CONFIG.COMET_SPAWN_INTERVAL_MIN +
        Math.random() *
          (CONFIG.COMET_SPAWN_INTERVAL_MAX - CONFIG.COMET_SPAWN_INTERVAL_MIN);

      if (this.time - this.lastCometTime > cometSpawnTime) {
        this.spawnCometBurst();
        this.lastCometTime = this.time;
      }

      this.renderer.render(this.scene, this.camera);
    };

    private setupEventListeners(): void {
      const handleResize = () => {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
      };

      window.addEventListener("resize", handleResize);
    }

    public destroy(): void {
      if (this.animationId) {
        cancelAnimationFrame(this.animationId);
      }

      this.renderer.dispose();
      this.starMesh.geometry.dispose();
      if (this.starMesh.material instanceof THREE.Material) {
        this.starMesh.material.dispose();
      }

      window.removeEventListener("resize", this.setupEventListeners);
    }
  }

  let starfieldContainer: HTMLDivElement;
  let starField3D: StarField3D | null = null;

  onMount(() => {
    try {
      starField3D = new StarField3D(starfieldContainer);
    } catch (error) {
      console.error("Failed to initialize 3D starfield:", error);
    }
  });

  onDestroy(() => {
    if (starField3D) {
      starField3D.destroy();
      starField3D = null;
    }
  });
</script>

<div
  bind:this={starfieldContainer}
  class="fixed inset-0 pointer-events-none"
  style="z-index: -1;"
></div>

<style>
  .fixed {
    position: fixed;
  }
  .inset-0 {
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
  }
  .pointer-events-none {
    pointer-events: none;
  }
</style>
