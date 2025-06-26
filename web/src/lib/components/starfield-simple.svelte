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
    renderer.setClearColor(new THREE.Color(0x000000), 1);
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
    scene.add(nearStars, farStars);

    const nebulaLayers: { mesh: THREE.Mesh; material: THREE.ShaderMaterial }[] =
      [];
    const layerData = [
      {
        radius: 250,
        scale: 0.008,
        speed: 0.008,
        brightness: 1.2, // Increased brightness
        color: 0x5a4a8c,
      },
      {
        radius: 500,
        scale: 0.004,
        speed: 0.005,
        brightness: 1.0, // Increased brightness
        color: 0x493c73,
      },
      {
        radius: 800,
        scale: 0.002,
        speed: 0.003,
        brightness: 0.8, // Increased brightness
        color: 0x382f5a,
      },
    ];

    layerData.forEach((data) => {
      const geometry = new THREE.SphereGeometry(data.radius, 32, 32);
      const material = new THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0.0 },
          scale: { value: data.scale },
          speed: { value: data.speed },
          brightness: { value: data.brightness },
          colorBright: { value: new THREE.Color(data.color) },
        },
        vertexShader: `varying vec3 vPosition; void main() { vPosition = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
        fragmentShader: `
          uniform float time; uniform float scale; uniform float speed; uniform float brightness; uniform vec3 colorBright; varying vec3 vPosition;
          vec4 p(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
          float n(vec3 P){vec3 Pi0=floor(P);vec3 Pi1=Pi0+vec3(1.0);Pi0=mod(Pi0,289.0);Pi1=mod(Pi1,289.0);vec3 Pf0=fract(P);vec4 ix=vec4(Pi0.x,Pi1.x,Pi0.x,Pi1.x);vec4 iy=vec4(Pi0.yy,Pi1.yy);vec4 ixy=p(p(ix)+iy);vec4 ixy0=p(ixy+Pi0.zzzz);vec4 ixy1=p(ixy+Pi1.zzzz);vec4 gx0=ixy0/7.0;vec4 gy0=fract(floor(gx0)/7.0)-0.5;gx0=fract(gx0);vec4 gz0=vec4(0.5)-abs(gx0)-abs(gy0);vec4 sz0=step(gz0,vec4(0.0));gx0-=sz0*(step(0.0,gx0)-0.5);gy0-=sz0*(step(0.0,gy0)-0.5);vec4 gx1=ixy1/7.0;vec4 gy1=fract(floor(gx1)/7.0)-0.5;gx1=fract(gx1);vec4 gz1=vec4(0.5)-abs(gx1)-abs(gy1);vec4 sz1=step(gz1,vec4(0.0));gx1-=sz1*(step(0.0,gx1)-0.5);gy1-=sz1*(step(0.0,gy1)-0.5);vec3 g000=vec3(gx0.x,gy0.x,gz0.x);vec3 g100=vec3(gx0.y,gy0.y,gz0.y);vec3 g010=vec3(gx0.z,gy0.z,gz0.z);vec3 g001=vec3(gx1.x,gy1.x,gz1.x);float n000=dot(g000,Pf0);float n100=dot(g100,vec3(Pf0.x-1.0,Pf0.yz));float n010=dot(g010,vec3(Pf0.x,Pf0.y-1.0,Pf0.z));float n001=dot(g001,vec3(Pf0.xy,Pf0.z-1.0));vec3 f=Pf0*Pf0*Pf0*(Pf0*(Pf0*6.0-15.0)+10.0);float n1=mix(n000,n100,f.x);float n2=mix(n010,dot(vec3(gx0.w,gy0.w,gz0.w),Pf0-vec3(1,1,0)),f.x);float n3=mix(n1,n2,f.y);float n4=mix(dot(g001,vec3(Pf0.xy,Pf0.z-1.0)),dot(vec3(gx1.y,gy1.y,gz1.y),Pf0-vec3(1,0,1)),f.x);float n5=mix(dot(vec3(gx1.z,gy1.z,gz1.z),Pf0-vec3(0,1,1)),dot(vec3(gx1.w,gy1.w,gz1.w),Pf0-vec3(1,1,1)),f.x);float n6=mix(n4,n5,f.y);return 2.2*mix(n3,n6,f.z);}
          float fbm(vec3 p) { float v = 0.0; float a = 0.5; for (int i = 0; i < 3; i++) { v += a * n(p); p *= 2.0; a *= 0.5; } return v; }
          void main() { vec3 pos = vPosition * scale + vec3(time * speed, 0.0, 0.0); float noise = (fbm(pos) + 1.0) * 0.5; float cn = pow(noise, 3.0); gl_FragColor = vec4(colorBright * cn, cn * brightness); }`,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
      });
      const mesh = new THREE.Mesh(geometry, material);
      nebulaLayers.push({ mesh, material });
      scene.add(mesh);
    });

    const clock = new THREE.Clock();

    const animate = () => {
      if (isDestroyed) return;
      animationFrameId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      nearStars.rotation.y = elapsedTime * 0.02;
      farStars.rotation.y = elapsedTime * 0.01;

      nebulaLayers.forEach((layer, i) => {
        layer.material.uniforms.time.value = elapsedTime;
        layer.mesh.rotation.y = elapsedTime * (0.002 / (i + 1));
      });

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

      nebulaLayers.forEach((layer) => {
        layer.mesh.geometry.dispose();
        layer.material.dispose();
      });

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
