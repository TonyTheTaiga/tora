<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as THREE from "three";

  // Configuration for maximum vibes
  const CONFIG = {
    STAR_COUNT: 3000,
    FIELD_SIZE: 1200,
    CAMERA_ROTATION_SPEED: 0.0003, // Slower, more meditative
    CAMERA_FLOAT_SPEED: 0.0001,
    CAMERA_BREATHING_SPEED: 0.0002,
    STAR_COLORS: [
      [0.9, 0.85, 0.7],   // Warm amber
      [0.8, 0.9, 1.0],    // Cool blue
      [1.0, 0.8, 0.9],    // Soft pink
      [0.9, 1.0, 0.8],    // Gentle green
      [1.0, 0.9, 0.7],    // Golden
      [0.7, 0.8, 1.0],    // Deep blue
    ],
  } as const;

  type Star = {
    position: THREE.Vector3;
    originalPosition: THREE.Vector3;
    color: THREE.Color;
    baseColor: THREE.Color;
    size: number;
    baseSize: number;
    brightness: number;
    baseBrightness: number;
    flickerSpeed: number;
    breathingPhase: number;
    wavePhase: number;
    layer: number; // 0 = far, 1 = mid, 2 = near
  };

  class StarField3D {
    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private renderer!: THREE.WebGLRenderer;
    private stars: Star[] = [];
    private starMesh!: THREE.Points;
    private nebulaMesh!: THREE.Mesh;
    private time: number = 0;
    private animationId: number | null = null;
    private baseFOV: number = 75;

    constructor(container: HTMLElement) {
      this.initializeScene(container);
      this.generateStars();
      this.createNebula();
      this.setupEventListeners();
      this.animate();
    }

    private initializeScene(container: HTMLElement): void {
      // Scene with subtle fog for depth
      this.scene = new THREE.Scene();
      this.scene.fog = new THREE.Fog(0x000011, 800, 2000);

      // Camera setup
      this.camera = new THREE.PerspectiveCamera(
        this.baseFOV,
        window.innerWidth / window.innerHeight,
        0.1,
        3000
      );
      this.camera.position.set(0, 0, 0);

      // Renderer with enhanced settings for beauty
      this.renderer = new THREE.WebGLRenderer({ 
        alpha: true, 
        antialias: true,
        powerPreference: "high-performance"
      });
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.renderer.setClearColor(0x000008, 1);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      
      container.appendChild(this.renderer.domElement);
    }

    private generateStars(): void {
      const positions = new Float32Array(CONFIG.STAR_COUNT * 3);
      const colors = new Float32Array(CONFIG.STAR_COUNT * 3);
      const sizes = new Float32Array(CONFIG.STAR_COUNT);

      for (let i = 0; i < CONFIG.STAR_COUNT; i++) {
        // Create stars in multiple layers for depth
        const layer = Math.floor(Math.random() * 3); // 0, 1, or 2
        const layerDistance = [1200, 800, 400][layer]; // Far, mid, near

        // Generate position in sphere at different distances
        const phi = Math.random() * Math.PI * 2;
        const costheta = Math.random() * 2 - 1;
        const theta = Math.acos(costheta);
        const radius = layerDistance * (0.5 + Math.random() * 0.5);

        const x = radius * Math.sin(theta) * Math.cos(phi);
        const y = radius * Math.sin(theta) * Math.sin(phi);
        const z = radius * Math.cos(theta);

        const position = new THREE.Vector3(x, y, z);
        const originalPosition = position.clone();

        // Enhanced star properties
        const colorIndex = Math.floor(Math.random() * CONFIG.STAR_COLORS.length);
        const [r, g, b] = CONFIG.STAR_COLORS[colorIndex];
        const baseColor = new THREE.Color(r, g, b);
        const color = baseColor.clone();
        
        const distance = position.length();
        const baseSize = (1.5 + Math.random() * 4.0) * (800 / distance) * (layer + 1);
        const baseBrightness = (0.3 + Math.random() * 0.7) * (layer * 0.3 + 0.4);

        this.stars.push({
          position,
          originalPosition,
          color,
          baseColor,
          size: baseSize,
          baseSize,
          brightness: baseBrightness,
          baseBrightness,
          flickerSpeed: 0.5 + Math.random() * 1.5,
          breathingPhase: Math.random() * Math.PI * 2,
          wavePhase: Math.random() * Math.PI * 2,
          layer,
        });

        // Set geometry arrays
        const i3 = i * 3;
        positions[i3] = x;
        positions[i3 + 1] = y;
        positions[i3 + 2] = z;
        
        colors[i3] = r;
        colors[i3 + 1] = g;
        colors[i3 + 2] = b;
        
        sizes[i] = baseSize;
      }

      // Enhanced shader material for hypnotic effects
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

      const material = new THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0.0 },
          breathingWave: { value: 0.0 },
          colorShift: { value: 0.0 },
        },
        vertexShader: `
          attribute float size;
          varying vec3 vColor;
          varying float vIntensity;
          uniform float time;
          uniform float breathingWave;
          uniform float colorShift;
          
          // Enhanced noise functions
          vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
          
          float snoise(vec2 v) {
            const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                               -0.577350269189626, 0.024390243902439);
            vec2 i  = floor(v + dot(v, C.yy) );
            vec2 x0 = v -   i + dot(i, C.xx);
            vec2 i1;
            i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
            vec4 x12 = x0.xyxy + C.xxzz;
            x12.xy -= i1;
            i = mod(i, 289.0);
            vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                            + i.x + vec3(0.0, i1.x, 1.0 ));
            vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
                                    dot(x12.zw,x12.zw)), 0.0);
            m = m*m ;
            m = m*m ;
            vec3 x = 2.0 * fract(p * C.www) - 1.0;
            vec3 h = abs(x) - 0.5;
            vec3 ox = floor(x + 0.5);
            vec3 a0 = x - ox;
            m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
            vec3 g;
            g.x  = a0.x  * x0.x  + h.x  * x0.y;
            g.yz = a0.yz * x12.xz + h.yz * x12.yw;
            return 130.0 * dot(m, g);
          }
          
          void main() {
            vColor = color;
            
            // Multiple layered effects
            float baseFlicker = snoise(vec2(time * 0.3 + position.x * 0.01, position.z * 0.01)) * 0.5 + 0.5;
            float breathingPulse = sin(time * 0.5 + breathingWave) * 0.3 + 0.7;
            float waveEffect = sin(time * 0.2 + position.y * 0.005) * 0.2 + 0.8;
            
            // Combine effects
            float finalIntensity = baseFlicker * breathingPulse * waveEffect;
            vIntensity = finalIntensity;
            
            float finalSize = size * (0.6 + finalIntensity * 0.8);
            
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = finalSize * (400.0 / -mvPosition.z);
            gl_Position = projectionMatrix * mvPosition;
          }
        `,
        fragmentShader: `
          varying vec3 vColor;
          varying float vIntensity;
          
          void main() {
            // Create soft, glowing stars
            vec2 center = gl_PointCoord - vec2(0.5);
            float distance = length(center);
            
            // Soft falloff with glow
            float alpha = 1.0 - smoothstep(0.0, 0.5, distance);
            float glow = exp(-distance * 4.0) * 0.3;
            
            // Color intensity based on effects
            vec3 finalColor = vColor * (vIntensity * 0.8 + 0.4);
            
            gl_FragColor = vec4(finalColor, (alpha + glow) * vIntensity);
          }
        `,
        transparent: true,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });

      this.starMesh = new THREE.Points(geometry, material);
      this.scene.add(this.starMesh);
    }

    private createNebula(): void {
      // Create subtle nebula background for atmosphere
      const nebulaGeometry = new THREE.SphereGeometry(2000, 32, 32);
      const nebulaMaterial = new THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0.0 },
          colorA: { value: new THREE.Color(0x001122) },
          colorB: { value: new THREE.Color(0x000408) },
        },
        vertexShader: `
          varying vec2 vUv;
          varying vec3 vPosition;
          
          void main() {
            vUv = uv;
            vPosition = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform float time;
          uniform vec3 colorA;
          uniform vec3 colorB;
          varying vec2 vUv;
          varying vec3 vPosition;
          
          float noise(vec3 p) {
            return fract(sin(dot(p, vec3(12.9898, 78.233, 54.53))) * 43758.5453);
          }
          
          void main() {
            vec3 pos = vPosition * 0.001;
            float n1 = noise(pos + time * 0.01);
            float n2 = noise(pos * 2.0 + time * 0.005);
            float n3 = noise(pos * 4.0 + time * 0.002);
            
            float combined = (n1 + n2 * 0.5 + n3 * 0.25) / 1.75;
            
            vec3 color = mix(colorA, colorB, combined);
            float alpha = combined * 0.1;
            
            gl_FragColor = vec4(color, alpha);
          }
        `,
        transparent: true,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      });

      this.nebulaMesh = new THREE.Mesh(nebulaGeometry, nebulaMaterial);
      this.scene.add(this.nebulaMesh);
    }

    private updateCamera(): void {
      const time = this.time;
      
      // Gentle spiral motion with floating sensation
      const rotationRadius = 100;
      const spiralSpeed = time * CONFIG.CAMERA_ROTATION_SPEED;
      const floatHeight = Math.sin(time * CONFIG.CAMERA_FLOAT_SPEED) * 50;
      const spiralHeight = Math.sin(time * CONFIG.CAMERA_FLOAT_SPEED * 0.7) * 30;
      
      this.camera.position.x = Math.cos(spiralSpeed) * rotationRadius;
      this.camera.position.z = Math.sin(spiralSpeed) * rotationRadius;
      this.camera.position.y = floatHeight + spiralHeight;
      
      // Breathing FOV effect
      const breathingFOV = this.baseFOV + Math.sin(time * CONFIG.CAMERA_BREATHING_SPEED) * 5;
      this.camera.fov = breathingFOV;
      this.camera.updateProjectionMatrix();
      
      // Smooth look-at with drift
      const lookTarget = new THREE.Vector3(
        Math.sin(spiralSpeed * 0.3) * 20,
        floatHeight * 0.3 + Math.sin(time * 0.0003) * 15,
        Math.cos(spiralSpeed * 0.3) * 20
      );
      this.camera.lookAt(lookTarget);
    }

    private updateEffects(): void {
      const time = this.time;
      
      // Update star shader uniforms
      if (this.starMesh.material instanceof THREE.ShaderMaterial) {
        this.starMesh.material.uniforms.time.value = time;
        this.starMesh.material.uniforms.breathingWave.value = Math.sin(time * 0.001) * Math.PI;
        this.starMesh.material.uniforms.colorShift.value = Math.sin(time * 0.0005) * 0.5 + 0.5;
      }

      // Update nebula
      if (this.nebulaMesh.material instanceof THREE.ShaderMaterial) {
        this.nebulaMesh.material.uniforms.time.value = time;
      }

      // Rotate nebula slowly
      this.nebulaMesh.rotation.y = time * 0.00005;
      this.nebulaMesh.rotation.x = Math.sin(time * 0.00003) * 0.1;
    }

    private animate = (): void => {
      this.animationId = requestAnimationFrame(this.animate);
      
      this.time = performance.now() * 0.001;

      this.updateCamera();
      this.updateEffects();

      this.renderer.render(this.scene, this.camera);
    };

    private setupEventListeners(): void {
      const handleResize = () => {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
      };

      window.addEventListener('resize', handleResize);
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
      if (this.nebulaMesh.material instanceof THREE.Material) {
        this.nebulaMesh.material.dispose();
      }
      
      window.removeEventListener('resize', this.setupEventListeners);
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