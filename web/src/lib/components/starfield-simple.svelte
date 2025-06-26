<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as THREE from "three";

  const NEAR_STAR_COUNT = 250;
  const FAR_STAR_COUNT = 500;
  const FIELD_WIDTH = 1000;
  const FIELD_DEPTH = 1000;

  let container: HTMLDivElement;
  let animationFrameId: number;
  let isDestroyed = false;

  onMount(() => {
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      FIELD_DEPTH * 2,
    );
    camera.position.z = FIELD_DEPTH;

    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    function createStars(count: number, size: number, color: number) {
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(count * 3);

      for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        positions[i3] = THREE.MathUtils.randFloatSpread(FIELD_WIDTH * 2);
        positions[i3 + 1] = THREE.MathUtils.randFloatSpread(FIELD_WIDTH * 2);
        positions[i3 + 2] = THREE.MathUtils.randFloatSpread(FIELD_DEPTH * 2);
      }

      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3),
      );

      const material = new THREE.PointsMaterial({
        size: size,
        color: color,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
      });

      return new THREE.Points(geometry, material);
    }

    const nearStars = createStars(NEAR_STAR_COUNT, 2.5, 0xbbd0ff);
    const farStars = createStars(FAR_STAR_COUNT, 1.5, 0x91a7ff);

    scene.add(nearStars);
    scene.add(farStars);

    const clock = new THREE.Clock();

    const animate = () => {
      if (isDestroyed) return;
      animationFrameId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Simple parallax rotation
      nearStars.rotation.y = elapsedTime * 0.02;
      farStars.rotation.y = elapsedTime * 0.01;

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

      nearStars.geometry.dispose();
      (nearStars.material as THREE.Material).dispose();
      farStars.geometry.dispose();
      (farStars.material as THREE.Material).dispose();

      renderer.dispose();
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
