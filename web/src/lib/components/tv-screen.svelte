<script lang="ts">
  import { onMount } from "svelte";
  import { browser } from "$app/environment";
  import * as THREE from "three";
  import { SVGLoader } from "three/examples/jsm/Addons.js";
  let container: HTMLDivElement;

  onMount(() => {
    if (!browser) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0, 0, 0);
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000,
    );
    camera.position.z = 5;
    const logoRenderer = new THREE.WebGLRenderer({
      antialias: true,
    });

    logoRenderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(logoRenderer.domElement);

    const logoLoader = new SVGLoader();
    const logoGroup = new THREE.Group();
    const typefaceGroup = new THREE.Group();
    scene.add(logoGroup);
    scene.add(typefaceGroup);

    const clock = new THREE.Clock();
    let velocity1 = new THREE.Vector2();
    let velocity2 = new THREE.Vector2();
    let logoWidth = 0;
    let logoHeight = 0;
    let typefaceWidth = 0;
    let typefaceHeight = 0;

    logoLoader.load("/favicon.svg", (data) => {
      logoGroup.scale.set(0.01, -0.01, 0.01);

      data.paths.forEach((path) => {
        const material = new THREE.MeshBasicMaterial({
          color: 0x89b4fa,
          side: THREE.DoubleSide,
          depthWrite: false,
        });

        SVGLoader.createShapes(path).forEach((shape) => {
          logoGroup.add(
            new THREE.Mesh(new THREE.ShapeGeometry(shape), material),
          );
        });
      });

      logoGroup.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(logoGroup);
      const size = box.getSize(new THREE.Vector3());
      logoGroup.updateMatrixWorld(true);
      logoWidth = size.x;
      logoHeight = size.y;
      const angle = Math.PI + Math.PI / 4;
      velocity1.set(Math.cos(angle), Math.sin(angle)).multiplyScalar(2);
    });

    logoLoader.load("/typeface.svg", (data) => {
      typefaceGroup.scale.set(0.005, -0.005, 0.005);

      data.paths.forEach((path) => {
        const material = new THREE.MeshBasicMaterial({
          color: 0x89b4fa,
          side: THREE.DoubleSide,
          depthWrite: false,
        });

        SVGLoader.createShapes(path).forEach((shape) => {
          typefaceGroup.add(
            new THREE.Mesh(new THREE.ShapeGeometry(shape), material),
          );
        });
      });

      typefaceGroup.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(typefaceGroup);
      const size = box.getSize(new THREE.Vector3());
      typefaceGroup.updateMatrixWorld(true);
      typefaceWidth = size.x;
      typefaceHeight = size.y;
      const angle = Math.PI - Math.PI / 4;
      velocity2.set(Math.cos(angle), Math.sin(angle)).multiplyScalar(2);
    });

    animate();

    function animate() {
      const dt = clock.getDelta();
      const dist = camera.position.z - logoGroup.position.z;

      logoGroup.position.x += velocity1.x * dt;
      logoGroup.position.y += velocity1.y * dt;
      const vHalf = THREE.MathUtils.degToRad(camera.fov * 0.5);
      const hHalf = Math.atan(Math.tan(vHalf) * camera.aspect);
      const halfV = Math.tan(vHalf) * dist;
      const halfH = Math.tan(hHalf) * dist;

      if (logoGroup.position.x + logoWidth > halfH) {
        velocity1.x = -Math.abs(velocity1.x);
        logoGroup.position.x = halfH - logoWidth;
      }
      if (logoGroup.position.x < -halfH) {
        velocity1.x = Math.abs(velocity1.x);
        logoGroup.position.x = -halfH;
      }
      if (logoGroup.position.y > halfV) {
        velocity1.y = -Math.abs(velocity1.y);
        logoGroup.position.y = halfV;
      }
      if (logoGroup.position.y - logoHeight < -halfV) {
        velocity1.y = Math.abs(velocity1.y);
        logoGroup.position.y = -halfV + logoHeight;
      }

      typefaceGroup.position.x += velocity2.x * dt;
      typefaceGroup.position.y += velocity2.y * dt;

      if (typefaceGroup.position.x + typefaceWidth > halfH) {
        velocity2.x = -Math.abs(velocity2.x);
        typefaceGroup.position.x = halfH - typefaceWidth;
      }
      if (typefaceGroup.position.x < -halfH) {
        velocity2.x = Math.abs(velocity2.x);
        typefaceGroup.position.x = -halfH;
      }

      if (typefaceGroup.position.y - typefaceHeight / 2 > halfV) {
        velocity2.y = -Math.abs(velocity2.y);
        typefaceGroup.position.y = halfV + typefaceHeight / 2;
      }

      if (typefaceGroup.position.y - typefaceHeight * 1.5 < -halfV) {
        velocity2.y = Math.abs(velocity2.y);
        typefaceGroup.position.y = -halfV + typefaceHeight * 1.5;
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
