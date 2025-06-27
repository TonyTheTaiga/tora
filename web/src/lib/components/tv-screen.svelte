<script lang="ts">
  import { onMount } from "svelte";
  import { browser } from "$app/environment";
  import * as THREE from "three";
  import { SVGLoader, SVGRenderer } from "three/examples/jsm/Addons.js";
  let container: HTMLDivElement;

  onMount(() => {
    if (!browser) {
      return;
    }

    THREE.ColorManagement.enabled = false;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0, 0, 0);
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000,
    );
    camera.position.z = 5;
    const logoRenderer = new SVGRenderer();
    logoRenderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(logoRenderer.domElement);

    const logoLoader = new SVGLoader();
    const group = new THREE.Group();
    scene.add(group);

    const clock = new THREE.Clock();
    let velocity = new THREE.Vector2();
    let logoWidth = 0;
    let logoHeight = 0;

    logoLoader.load("/favicon.svg", (data) => {
      group.scale.set(0.01, -0.01, 0.01);

      data.paths.forEach((path) => {
        const material = new THREE.MeshBasicMaterial({
          color: 0x89b4fa,
          side: THREE.DoubleSide,
          depthWrite: false,
        });

        SVGLoader.createShapes(path).forEach((shape) => {
          group.add(new THREE.Mesh(new THREE.ShapeGeometry(shape), material));
        });
      });

      group.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(group);
      const size = box.getSize(new THREE.Vector3());
      group.updateMatrixWorld(true);
      logoWidth = size.x;
      logoHeight = size.y;
      const angle = Math.PI + Math.PI / 4;
      velocity.set(Math.cos(angle), Math.sin(angle)).multiplyScalar(2);
    });

    animate();

    function animate() {
      const dt = clock.getDelta();
      const dist = camera.position.z - group.position.z;

      group.position.x += velocity.x * dt;
      group.position.y += velocity.y * dt;

      const vHalf = THREE.MathUtils.degToRad(camera.fov * 0.5);
      const hHalf = Math.atan(Math.tan(vHalf) * camera.aspect);
      const halfV = Math.tan(vHalf) * dist;
      const halfH = Math.tan(hHalf) * dist;

      if (group.position.x + logoWidth > halfH) {
        velocity.x = -Math.abs(velocity.x);
        group.position.x = halfH - logoWidth;
      }
      if (group.position.x < -halfH) {
        velocity.x = Math.abs(velocity.x);
        group.position.x = -halfH;
      }
      if (group.position.y > halfV) {
        velocity.y = -Math.abs(velocity.y);
        group.position.y = halfV;
      }
      if (group.position.y - logoHeight < -halfV) {
        velocity.y = Math.abs(velocity.y);
        group.position.y = -halfV + logoHeight;
      }

      logoRenderer.render(scene, camera);
      requestAnimationFrame(animate);
    }

    function onResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      logoRenderer.setSize(window.innerWidth, window.innerHeight);
    }

    window.addEventListener("resize", onResize, false);

    return () => {
      window.removeEventListener("resize", onResize, false);
    };
  });
</script>

<div
  bind:this={container}
  class="fixed inset-0 -z-10 pointer-events-none w-full"
></div>
