<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import type p5 from "p5";
  import { KNOWN_STARS } from "./star-catalog";

  export let latitude: number = 40.7128;
  export let longitude: number = -74.006;
  export let magnitudeCutoff: number = 4.5;
  export let minStarCount: number = 350;

  const StarType = {
    DISTANT: "distant",
    MEDIUM: "bright",
    BRIGHT: "brighter",
    SUPERBRIGHT: "superbright",
  } as const;
  type StarType = (typeof StarType)[keyof typeof StarType];

  const StarSize = {
    TINY: 2,
    SMALL: 3,
    MEDIUM: 4,
    LARGE: 5,
    MASSIVE: 6,
  } as const;
  type StarSize = (typeof StarSize)[keyof typeof StarSize];

  const StarShape = {
    CIRCLE: "circle",
    DIAMOND: "diamond",
    PLUS: "plus",
    STAR: "star",
    TWINKLE: "twinkle",
  } as const;
  type StarShape = (typeof StarShape)[keyof typeof StarShape];

  type StarProps = {
    id: number;
    name: string;
    ra: number;
    dec: number;
    mag: number;
    x: number;
    y: number;
    size: StarSize;
    maxBrightness: number;
    flickerSpeed: number;
    noiseOffset: number;
    color: readonly [number, number, number];
    type: StarType;
    shape: StarShape;
    isStatic: boolean;
  };

  type CometProps = {
    id: number;
    startX: number;
    startY: number;
    endX: number;
    endY: number;
    speed: number;
    tailLength: number;
    brightness: number;
    color: readonly [number, number, number];
  };

  const CONFIG = {
    STAR_COLORS: [
      [245, 240, 225],
      [255, 245, 215],
      [255, 235, 195],
      [255, 215, 175],
      [170, 205, 255],
      [255, 180, 150],
    ],
    RECALCULATION_INTERVAL: 600,
    COMET_SPAWN_INTERVAL_MIN: 3,
    COMET_SPAWN_INTERVAL_MAX: 10,
    MAX_COMETS_AT_ONCE: 5,
    AZIMUTH_FOV: 120,
    TWINKLE_EFFECT: {
      MULTI_NOISE_FACTOR: 0.7,
      SIZE_FLICKER_AMOUNT: 0.8,
      SPIKE_CHANCE: 0.001,
      SPIKE_INTENSITY: 0.4,
    },
  } as const;

  function easeInOutQuad(t: number): number {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
  }

  function getRandomShape(isBright: boolean): StarShape {
    const baseShapes = [StarShape.CIRCLE, StarShape.DIAMOND, StarShape.PLUS];
    if (!isBright) {
      return baseShapes[Math.floor(Math.random() * baseShapes.length)];
    }
    const brightShapes = [...baseShapes, StarShape.STAR, StarShape.TWINKLE];
    return brightShapes[Math.floor(Math.random() * brightShapes.length)];
  }

  function calculateFlickerSpeed(type: StarType): number {
    const speeds = {
      [StarType.DISTANT]: { min: 1.0, range: 2.0 },
      [StarType.MEDIUM]: { min: 1.5, range: 2.5 },
      [StarType.BRIGHT]: { min: 2.0, range: 3.0 },
      [StarType.SUPERBRIGHT]: { min: 2.5, range: 3.5 },
    };
    const s = speeds[type] || speeds[StarType.DISTANT];
    return s.min + Math.random() * s.range;
  }

  function determineStarColorFromSpectralType(
    spectralType: string,
  ): readonly [number, number, number] {
    if (!spectralType) return CONFIG.STAR_COLORS[0];
    const typeChar = spectralType.charAt(0).toUpperCase();
    const colorMap: Record<string, readonly [number, number, number]> = {
      O: CONFIG.STAR_COLORS[4],
      B: CONFIG.STAR_COLORS[4],
      A: CONFIG.STAR_COLORS[0],
      F: CONFIG.STAR_COLORS[1],
      G: CONFIG.STAR_COLORS[2],
      K: CONFIG.STAR_COLORS[3],
      M: CONFIG.STAR_COLORS[5],
    };
    return colorMap[typeChar] || CONFIG.STAR_COLORS[0];
  }

  function dateToJulianDate(date: Date): number {
    return date.getTime() / 86400000 + 2440587.5;
  }

  function calculateLST(jd: number, lon: number): number {
    const T = (jd - 2451545.0) / 36525.0;
    let gmst =
      280.46061837 +
      360.98564736629 * (jd - 2451545.0) +
      0.000387933 * T * T -
      (T * T * T) / 38710000;
    gmst = gmst % 360;
    if (gmst < 0) gmst += 360;
    const lst = gmst + lon;
    return lst / 15;
  }

  class Star {
    readonly id: number;
    x: number;
    y: number;
    readonly size: StarSize;
    brightness: number;
    readonly maxBrightness: number;
    readonly flickerSpeed: number;
    readonly noiseOffset: number;
    readonly color: readonly [number, number, number];
    readonly type: StarType;
    readonly shape: StarShape;
    isStatic: boolean;
    readonly name: string;
    readonly mag: number;
    readonly ra: number;
    readonly dec: number;

    currentSize: number;
    spikeBrightness: number;

    constructor(props: StarProps) {
      this.id = props.id;
      this.x = props.x;
      this.y = props.y;
      this.size = props.size;
      this.maxBrightness = props.maxBrightness;
      this.flickerSpeed = props.flickerSpeed;
      this.noiseOffset = props.noiseOffset;
      this.color = props.color;
      this.type = props.type;
      this.shape = props.shape;
      this.isStatic = props.isStatic;
      this.name = props.name;
      this.mag = props.mag;
      this.ra = props.ra;
      this.dec = props.dec;

      this.currentSize = this.size;
      this.spikeBrightness = 0;
      this.brightness = this.isStatic
        ? this.maxBrightness
        : this.maxBrightness * (0.5 + Math.random() * 0.5);
    }

    update(p: p5, time: number): void {
      if (this.isStatic) return;

      const primaryNoise = p.noise(time * this.flickerSpeed + this.noiseOffset);

      const secondaryNoise = p.noise(
        time * this.flickerSpeed * 3 + this.noiseOffset + 1000,
      );

      const combinedNoise = p.lerp(
        primaryNoise,
        secondaryNoise,
        CONFIG.TWINKLE_EFFECT.MULTI_NOISE_FACTOR,
      );

      this.spikeBrightness *= 0.7;
      if (Math.random() < CONFIG.TWINKLE_EFFECT.SPIKE_CHANCE * 10) {
        this.spikeBrightness = CONFIG.TWINKLE_EFFECT.SPIKE_INTENSITY * 2;
      }

      const flickerValue = p.constrain(
        combinedNoise + this.spikeBrightness,
        0,
        1,
      );

      const brightnessRange = 0.9;
      const minBrightnessFactor = 1.0 - brightnessRange;
      this.brightness =
        this.maxBrightness *
        (minBrightnessFactor + flickerValue * brightnessRange);

      const sizeFlicker = (flickerValue - 0.5) * 2;
      this.currentSize =
        this.size + sizeFlicker * CONFIG.TWINKLE_EFFECT.SIZE_FLICKER_AMOUNT * 2;
      this.currentSize = Math.max(1, this.currentSize);
    }

    render(p: p5): void {
      const [r, g, b] = this.color;
      p.fill(r, g, b, this.brightness);
      p.noStroke();

      const size = this.currentSize;
      const halfSize = size / 2;

      switch (this.shape) {
        case StarShape.CIRCLE:
          p.circle(this.x, this.y, size);
          break;
        case StarShape.DIAMOND:
          p.beginShape();
          p.vertex(this.x, this.y - halfSize);
          p.vertex(this.x + halfSize, this.y);
          p.vertex(this.x, this.y + halfSize);
          p.vertex(this.x - halfSize, this.y);
          p.endShape(p.CLOSE);
          break;
        case StarShape.PLUS:
          p.rect(this.x - halfSize, this.y - 0.5, size, 1);
          p.rect(this.x - 0.5, this.y - halfSize, 1, size);
          break;
        case StarShape.STAR:
          this.drawStarShape(p, this.x, this.y, size * 0.6, size * 0.3);
          break;
        case StarShape.TWINKLE:
          p.rect(this.x - halfSize, this.y - 0.5, size, 1);
          p.rect(this.x - 0.5, this.y - halfSize, 1, size);
          const d = size * 0.7;
          p.stroke(r, g, b, this.brightness);
          p.strokeWeight(1);
          p.line(
            this.x - d / 2,
            this.y - d / 2,
            this.x + d / 2,
            this.y + d / 2,
          );
          p.line(
            this.x - d / 2,
            this.y + d / 2,
            this.x + d / 2,
            this.y - d / 2,
          );
          p.noStroke();
          break;
      }

      if (size >= StarSize.MEDIUM && this.brightness > 120) {
        p.blendMode(p.ADD);
        p.fill(r, g, b, this.brightness * 0.2);
        p.circle(this.x, this.y, size * 2);
        p.blendMode(p.BLEND);
      }
      if (this.type === StarType.SUPERBRIGHT && this.brightness > 140) {
        p.blendMode(p.ADD);
        p.fill(r, g, b, this.brightness * 0.1);
        p.circle(this.x, this.y, size * 3);
        p.blendMode(p.BLEND);
      }
    }

    private drawStarShape(
      p: p5,
      x: number,
      y: number,
      outerRadius: number,
      innerRadius: number,
    ): void {
      const angleStep = p.PI / 5;
      p.beginShape();
      for (let i = 0; i < 10; i++) {
        const angle = i * angleStep - p.PI / 2;
        const radius = i % 2 === 0 ? outerRadius : innerRadius;
        p.vertex(x + p.cos(angle) * radius, y + p.sin(angle) * radius);
      }
      p.endShape(p.CLOSE);
    }
  }

  class Comet {
    readonly id: number;
    x: number;
    y: number;
    readonly startX: number;
    readonly startY: number;
    readonly endX: number;
    readonly endY: number;
    readonly speed: number;
    readonly tailLength: number;
    readonly brightness: number;
    readonly color: readonly [number, number, number];
    progress: number;
    active: boolean;

    constructor(props: CometProps) {
      this.id = props.id;
      this.startX = props.startX;
      this.startY = props.startY;
      this.endX = props.endX;
      this.endY = props.endY;
      this.speed = props.speed;
      this.tailLength = props.tailLength;
      this.brightness = props.brightness;
      this.color = props.color;
      this.x = this.startX;
      this.y = this.startY;
      this.progress = 0;
      this.active = true;
    }

    update(deltaTime: number): void {
      if (!this.active) return;
      this.progress += deltaTime * this.speed;
      if (this.progress >= 1) {
        this.active = false;
        return;
      }
      const easedProgress = easeInOutQuad(this.progress);
      this.x = this.startX + (this.endX - this.startX) * easedProgress;
      this.y = this.startY + (this.endY - this.startY) * easedProgress;
    }

    render(p: p5): void {
      const [r, g, b] = this.color;
      const dx = this.endX - this.startX;
      const dy = this.endY - this.startY;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const tailX = -(dx / dist) * this.tailLength;
      const tailY = -(dy / dist) * this.tailLength;
      for (let i = 0; i < 8; i++) {
        const t = i / 8;
        p.stroke(r, g, b, this.brightness * (1 - t) * 0.6);
        p.strokeWeight((1 - t) * 3 + 1);
        p.line(
          this.x + tailX * t,
          this.y + tailY * t,
          this.x + tailX * (t + 1 / 8),
          this.y + tailY * (t + 1 / 8),
        );
      }
      p.blendMode(p.ADD);
      p.noStroke();
      p.fill(r, g, b, this.brightness);
      p.circle(this.x, this.y, 4);
      p.fill(255, 255, 255, this.brightness * 0.8);
      p.circle(this.x, this.y, 2);
      p.blendMode(p.BLEND);
    }
  }

  type StarFieldConfig = {
    latitude: number;
    longitude: number;
    magnitudeCutoff: number;
    minStarCount: number;
  };

  class StarField {
    private p: p5;
    private config: StarFieldConfig;
    private stars: Star[] = [];
    private comets: Comet[] = [];
    private time = 0;
    private lastCometTime = 0;
    private nextRecalcTime = 0;
    private staticStarsBuffer: p5.Graphics | null = null;
    private width: number;
    private height: number;

    constructor(p: p5, config: StarFieldConfig) {
      this.p = p;
      this.config = config;
      this.width = p.windowWidth;
      this.height = p.windowHeight;
      this.regenerate();
    }

    public updateConfig(newConfig: StarFieldConfig) {
      this.config = { ...this.config, ...newConfig };
    }

    public regenerate(): void {
      this.stars = [];
      this.comets = [];
      this.time = 0;
      this.nextRecalcTime = CONFIG.RECALCULATION_INTERVAL;
      this.generateRealStarField();
    }

    private generateRealStarField(): void {
      const now = new Date();
      const jd = dateToJulianDate(now);
      const lst = calculateLST(jd, this.config.longitude);
      let starIdCounter = 0;
      for (const starData of KNOWN_STARS) {
        if (starData.mag > this.config.magnitudeCutoff) continue;
        const hourAngle = (lst * 15 - starData.ra * 15 + 360) % 360;
        const latRad = (this.config.latitude * Math.PI) / 180;
        const decRad = (starData.dec * Math.PI) / 180;
        const haRad = (hourAngle * Math.PI) / 180;
        const sinAlt =
          Math.sin(decRad) * Math.sin(latRad) +
          Math.cos(decRad) * Math.cos(latRad) * Math.cos(haRad);
        const alt = Math.asin(sinAlt);
        if (alt < 0) continue;
        const cosA =
          (Math.sin(decRad) - Math.sin(alt) * Math.sin(latRad)) /
          (Math.cos(alt) * Math.cos(latRad));
        let az = Math.acos(this.p.constrain(cosA, -1, 1));
        if (Math.sin(haRad) > 0) az = 2 * Math.PI - az;
        const { x, y } = this.projectHorizonToScreen(
          az * (180 / Math.PI),
          alt * (180 / Math.PI),
        );
        const type = this.determineStarTypeFromMagnitude(starData.mag);
        const isStatic = starData.mag >= 3.5 || type === StarType.DISTANT;
        this.stars.push(
          new Star({
            id: starIdCounter++,
            name: starData.name,
            ra: starData.ra,
            dec: starData.dec,
            mag: starData.mag,
            x,
            y,
            size: this.mapMagnitudeToSize(starData.mag),
            maxBrightness: this.mapMagnitudeToBrightness(starData.mag),
            flickerSpeed: calculateFlickerSpeed(type),
            noiseOffset: Math.random() * 1000,
            color: determineStarColorFromSpectralType(starData.spectralType),
            type,
            shape: getRandomShape(!isStatic),
            isStatic,
          }),
        );
      }
      const numExtraStars = this.config.minStarCount - this.stars.length;
      for (let i = 0; i < numExtraStars; i++) {
        this.stars.push(
          new Star({
            id: starIdCounter++,
            name: "Distant Star",
            ra: 0,
            dec: 0,
            mag: 5.0 + Math.random(),
            x: Math.random() * this.width,
            y: Math.random() * this.height,
            size: StarSize.TINY,
            maxBrightness: this.mapMagnitudeToBrightness(
              5.5 + Math.random() * 0.5,
            ),
            flickerSpeed: calculateFlickerSpeed(StarType.DISTANT),
            noiseOffset: Math.random() * 1000,
            color: CONFIG.STAR_COLORS[0],
            type: StarType.DISTANT,
            shape: StarShape.CIRCLE,
            isStatic: true,
          }),
        );
      }
      this.stars.sort((a, b) => a.maxBrightness - b.maxBrightness);
      this.renderStaticStarsToBuffer();
    }

    private projectHorizonToScreen(
      az: number,
      alt: number,
    ): { x: number; y: number } {
      const y = this.p.map(alt, 0, 90, this.height * 0.8, this.height * 0.1);
      let azDiff = az - 0;
      if (azDiff > 180) azDiff -= 360;
      if (azDiff < -180) azDiff += 360;
      const x = this.p.map(
        azDiff,
        -CONFIG.AZIMUTH_FOV / 2,
        CONFIG.AZIMUTH_FOV / 2,
        0,
        this.width,
      );
      return { x, y };
    }

    private mapMagnitudeToSize = (mag: number): StarSize => {
      if (mag < -0.5) return StarSize.MASSIVE;
      if (mag < 0.5) return StarSize.LARGE;
      if (mag < 1.5) return StarSize.MEDIUM;
      if (mag < 2.5) return StarSize.SMALL;
      return StarSize.TINY;
    };
    private mapMagnitudeToBrightness = (mag: number): number =>
      this.p.map(
        this.p.constrain(mag, -1.5, this.config.magnitudeCutoff),
        this.config.magnitudeCutoff,
        -1.5,
        80,
        255,
      );
    private determineStarTypeFromMagnitude = (mag: number): StarType => {
      if (mag < 0) return StarType.SUPERBRIGHT;
      if (mag < 1.0) return StarType.BRIGHT;
      if (mag < 2.5) return StarType.MEDIUM;
      return StarType.DISTANT;
    };

    private renderStaticStarsToBuffer(): void {
      if (this.staticStarsBuffer) this.staticStarsBuffer.remove();
      this.staticStarsBuffer = this.p.createGraphics(this.width, this.height);
      this.staticStarsBuffer.clear();
      this.staticStarsBuffer.noStroke();
      for (const star of this.stars) {
        if (star.isStatic) {
          const [r, g, b] = star.color;
          this.staticStarsBuffer.fill(r, g, b, star.brightness);
          this.staticStarsBuffer.circle(star.x, star.y, star.size);
        }
      }
    }

    public update(deltaTime: number): void {
      this.time += deltaTime;
      for (const star of this.stars) star.update(this.p, this.time);
      for (let i = this.comets.length - 1; i >= 0; i--) {
        this.comets[i].update(deltaTime);
        if (!this.comets[i].active) this.comets.splice(i, 1);
      }
      if (this.time >= this.nextRecalcTime) this.regenerate();
      const cometSpawnTime =
        CONFIG.COMET_SPAWN_INTERVAL_MIN +
        Math.random() *
          (CONFIG.COMET_SPAWN_INTERVAL_MAX - CONFIG.COMET_SPAWN_INTERVAL_MIN);
      if (this.time - this.lastCometTime > cometSpawnTime) {
        this.spawnCometBurst();
        this.lastCometTime = this.time;
      }
    }

    private spawnCometBurst(): void {
      const numComets =
        Math.floor(Math.random() * CONFIG.MAX_COMETS_AT_ONCE) + 1;

      for (let i = 0; i < numComets; i++) {
        this.spawnComet();
      }
    }

    private spawnComet(): void {
      let startX, startY, endX, endY;
      const edge = Math.floor(Math.random() * 4);
      switch (edge) {
        case 0:
          startX = Math.random() * this.width;
          startY = -50;
          endX = Math.random() * this.width;
          endY = this.height + 50;
          break;
        case 1:
          startX = this.width + 50;
          startY = Math.random() * this.height;
          endX = -50;
          endY = Math.random() * this.height;
          break;
        case 2:
          startX = Math.random() * this.width;
          startY = this.height + 50;
          endX = Math.random() * this.width;
          endY = -50;
          break;
        default:
          startX = -50;
          startY = Math.random() * this.height;
          endX = this.width + 50;
          endY = Math.random() * this.height;
          break;
      }
      this.comets.push(
        new Comet({
          id: this.comets.length,
          startX,
          startY,
          endX,
          endY,
          speed: 0.2 + Math.random() * 0.3,
          tailLength: 50 + Math.random() * 80,
          brightness: 180 + Math.random() * 75,
          color:
            CONFIG.STAR_COLORS[
              Math.floor(Math.random() * CONFIG.STAR_COLORS.length)
            ],
        }),
      );
    }

    public render(): void {
      this.p.clear();
      if (this.staticStarsBuffer) this.p.image(this.staticStarsBuffer, 0, 0);
      for (const star of this.stars) {
        if (!star.isStatic) star.render(this.p);
      }
      for (const comet of this.comets) {
        if (comet.active) comet.render(this.p);
      }
    }

    public resize(width: number, height: number): void {
      this.width = width;
      this.height = height;
      this.regenerate();
    }
  }

  let starfieldContainer: HTMLDivElement;
  let p5Instance: p5 | null = null;
  let starField: StarField | null = null;

  $: if (p5Instance && starField) {
    starField.updateConfig({
      latitude,
      longitude,
      magnitudeCutoff,
      minStarCount,
    });
    starField.regenerate();
  }

  onMount(async () => {
    try {
      const p5Module = await import("p5");
      const p5 = p5Module.default;
      const starfieldSketch = (p: p5) => {
        let lastFrameTime = 0;
        p.setup = () => {
          const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
          canvas.parent(starfieldContainer);
          canvas.style("position", "fixed");
          canvas.style("top", "0");
          canvas.style("left", "0");
          canvas.style("z-index", "-1");
          canvas.style("pointer-events", "none");
          starField = new StarField(p, {
            latitude,
            longitude,
            magnitudeCutoff,
            minStarCount,
          });
          lastFrameTime = p.millis();
        };
        p.draw = () => {
          const currentTime = p.millis();
          const deltaTime = (currentTime - lastFrameTime) / 1000;
          lastFrameTime = currentTime;
          starField?.update(deltaTime);
          starField?.render();
        };
        p.windowResized = () => {
          p.resizeCanvas(p.windowWidth, p.windowHeight);
          starField?.resize(p.windowWidth, p.windowHeight);
        };
      };
      p5Instance = new p5(starfieldSketch);
    } catch (error) {
      console.error("Failed to load starfield:", error);
    }
  });

  onDestroy(() => {
    if (p5Instance) {
      p5Instance.remove();
      p5Instance = null;
      starField = null;
    }
  });
</script>

<div
  bind:this={starfieldContainer}
  class="fixed inset-0 pointer-events-none"
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
