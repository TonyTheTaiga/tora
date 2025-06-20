<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as THREE from "three";

  const STAR_COUNT = 2000;
  const FIELD_DEPTH = 1000;

  let container: HTMLDivElement;
  let animationFrameId: number;
  let isDestroyed = false;

  onMount(() => {
    if (!container) return;

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      FIELD_DEPTH + 1,
    );
    camera.position.z = 1;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(new THREE.Color(0x000000), 1);
    container.appendChild(renderer.domElement);

    const positions = new Float32Array(STAR_COUNT * 3);
    const colors = new Float32Array(STAR_COUNT * 3);
    const sizes = new Float32Array(STAR_COUNT);
    const randoms = new Float32Array(STAR_COUNT);

    const starColors = [
      new THREE.Color(0x9bb0ff),
      new THREE.Color(0xaabfff),
      new THREE.Color(0xcad7ff),
      new THREE.Color(0xf8f7ff),
    ];

    for (let i = 0; i < STAR_COUNT; i++) {
      const i3 = i * 3;

      positions[i3] = (Math.random() - 0.5) * 1000;
      positions[i3 + 1] = (Math.random() - 0.5) * 1000;
      positions[i3 + 2] = -Math.random() * FIELD_DEPTH;

      const color = starColors[Math.floor(Math.random() * starColors.length)];
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;

      sizes[i] = Math.random() * 2.0 + 1.5;
      randoms[i] = Math.random() * 10.0;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute("random", new THREE.BufferAttribute(randoms, 1));

    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0.0 },
      },
      vertexShader: `
        attribute float size;
        attribute float random;
        varying vec3 vColor;
        varying float vRandom;

        void main() {
          vColor = color;
          vRandom = random;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform float time;
        varying vec3 vColor;
        varying float vRandom;

        void main() {
          float dist = length(gl_PointCoord - vec2(0.5));
          if (dist > 0.5) discard;

          float blink = 0.7 + sin(time * 0.1 + vRandom) * 0.3;
          float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
          
          gl_FragColor = vec4(vColor, alpha * blink);
        }
      `,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      transparent: true,
      vertexColors: true,
    });

    const starField = new THREE.Points(geometry, material);
    scene.add(starField);

    const clock = new THREE.Clock();

    const animate = () => {
      if (isDestroyed) return;
      animationFrameId = requestAnimationFrame(animate);

      const elapsedTime = clock.getElapsedTime();
      material.uniforms.time.value = elapsedTime;
      starField.rotation.y = elapsedTime * 0.003;
      starField.rotation.x = elapsedTime * 0.001;

      renderer.render(scene, camera);
    };

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener("resize", handleResize);
    animate();

    onDestroy(() => {
      isDestroyed = true;
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      if (container && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    });
  });
</script>

<div
  bind:this={container}
  class="fixed inset-0 -z-10 pointer-events-none"
></div>

<style>
  div {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
    z-index: -1;
    pointer-events: none;
    background-color: #000;
  }
</style>
