<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import type p5 from "p5";
  import { BRIGHT_STARS_CATALOG } from "./star-catalog";

  let starfieldContainer: HTMLDivElement;
  let p5Instance: p5 | null = null;

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

  const STAR_COLORS: readonly [number, number, number][] = [
    [245, 240, 225],
    [255, 245, 215],
    [255, 235, 195],
    [255, 215, 175],
    [170, 205, 255],
    [255, 180, 150],
  ] as const;

  const NEW_YORK_LAT = 40.7128;
  const NEW_YORK_LON = -74.006;
  const NYC_MAGNITUDE_CUTOFF = 4.5;

  function easeInOutQuad(t: number): number {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
  }

  function getRandomShape(isBright: boolean): StarShape {
    if (isBright) {
      const shapes = [
        StarShape.CIRCLE,
        StarShape.DIAMOND,
        StarShape.PLUS,
        StarShape.STAR,
        StarShape.TWINKLE,
      ];
      return shapes[Math.floor(Math.random() * shapes.length)];
    } else {
      const shapes = [StarShape.CIRCLE, StarShape.DIAMOND, StarShape.PLUS];
      return shapes[Math.floor(Math.random() * shapes.length)];
    }
  }

  function calculateFlickerSpeed(type: StarType): number {
    switch (type) {
      case StarType.DISTANT:
        return 0.2 + Math.random() * 0.4;
      case StarType.MEDIUM:
        return 0.15 + Math.random() * 0.35;
      case StarType.BRIGHT:
        return 0.1 + Math.random() * 0.3;
      case StarType.SUPERBRIGHT:
        return 0.08 + Math.random() * 0.25;
      default:
        return 0.2;
    }
  }

  function determineStarColorFromSpectralType(
    spectralType: string,
  ): readonly [number, number, number] {
    if (!spectralType) return STAR_COLORS[0];

    const typeChar = spectralType.charAt(0).toUpperCase();
    switch (typeChar) {
      case "O":
        return STAR_COLORS[4];
      case "B":
        return STAR_COLORS[4];
      case "A":
        return STAR_COLORS[0];
      case "F":
        return STAR_COLORS[1];
      case "G":
        return STAR_COLORS[2];
      case "K":
        return STAR_COLORS[3];
      case "M":
        return STAR_COLORS[5];
      default:
        return STAR_COLORS[0];
    }
  }

  class Star {
    readonly id: number;
    x: number;
    y: number;
    size: StarSize;
    brightness: number;
    readonly maxBrightness: number;
    readonly flickerSpeed: number;
    readonly noiseOffset: number;
    readonly color: readonly [number, number, number];
    readonly type: StarType;
    readonly shape: StarShape;
    twinklePhase: number;
    pulsePhase: number;
    isStatic: boolean;

    private p: p5;
    readonly name: string;
    readonly mag: number;
    readonly ra: number;
    readonly dec: number;

    constructor(
      p: p5,
      props: {
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
      },
    ) {
      this.p = p;
      this.id = props.id;
      this.name = props.name;
      this.ra = props.ra;
      this.dec = props.dec;
      this.mag = props.mag;
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
      this.twinklePhase = Math.random() * this.p.TWO_PI;
      this.pulsePhase = Math.random() * this.p.TWO_PI;

      if (this.isStatic) {
        this.brightness = this.maxBrightness;
      } else {
        this.brightness = this.maxBrightness * (0.5 + Math.random() * 0.5);
      }
    }

    update(time: number, deltaTime: number): void {
      if (this.isStatic) return;

      const noiseVal = this.p.noise(
        time * this.flickerSpeed * 0.1 + this.noiseOffset,
      );
      const primaryFlicker = this.p.map(noiseVal, 0, 1, -1, 1);

      this.pulsePhase += deltaTime * this.flickerSpeed * 0.5;
      const pulseIntensity =
        this.size >= StarSize.MEDIUM ? Math.sin(this.pulsePhase) * 0.15 : 0;

      const combinedFlicker = primaryFlicker + pulseIntensity;
      const normalizedFlicker = this.p.constrain(
        (combinedFlicker + 1) * 0.5,
        0.2,
        1,
      );

      let brightnessModifier = 1;
      switch (this.type) {
        case StarType.DISTANT:
          brightnessModifier = 0.6 + normalizedFlicker * 0.4;
          break;
        case StarType.MEDIUM:
          brightnessModifier = 0.5 + normalizedFlicker * 0.5;
          break;
        case StarType.BRIGHT:
          brightnessModifier = 0.4 + normalizedFlicker * 0.6;
          break;
        case StarType.SUPERBRIGHT:
          brightnessModifier = 0.6 + normalizedFlicker * 0.4;
          break;
      }
      this.brightness = this.maxBrightness * brightnessModifier;
    }

    render(): void {
      const [r, g, b] = this.color;
      const pixelX = Math.floor(this.x);
      const pixelY = Math.floor(this.y);
      const size = this.size;
      const halfSize = Math.floor(size / 2);

      this.p.fill(r, g, b, this.brightness);
      this.p.noStroke();

      switch (this.shape) {
        case StarShape.CIRCLE:
          this.p.circle(this.x, this.y, size);
          break;
        case StarShape.DIAMOND:
          this.p.beginShape();
          this.p.vertex(this.x, this.y - halfSize);
          this.p.vertex(this.x + halfSize, this.y);
          this.p.vertex(this.x, this.y + halfSize);
          this.p.vertex(this.x - halfSize, this.y);
          this.p.endShape(this.p.CLOSE);
          break;
        case StarShape.PLUS:
          this.p.rect(pixelX - halfSize, pixelY - 0.5, size, 1);
          this.p.rect(pixelX - 0.5, pixelY - halfSize, 1, size);
          break;
        case StarShape.STAR:
          this.drawStarShape(this.x, this.y, size * 0.6, size * 0.3);
          break;
        case StarShape.TWINKLE:
          this.p.rect(pixelX - halfSize, pixelY - 0.5, size, 1);
          this.p.rect(pixelX - 0.5, pixelY - halfSize, 1, size);
          const diagonalLength = size * 0.7;
          this.p.stroke(r, g, b, this.brightness);
          this.p.strokeWeight(1);
          this.p.line(
            this.x - diagonalLength / 2,
            this.y - diagonalLength / 2,
            this.x + diagonalLength / 2,
            this.y + diagonalLength / 2,
          );
          this.p.line(
            this.x - diagonalLength / 2,
            this.y + diagonalLength / 2,
            this.x + diagonalLength / 2,
            this.y - diagonalLength / 2,
          );
          this.p.noStroke();
          break;
      }

      if (this.size >= StarSize.MEDIUM && this.brightness > 120) {
        this.p.blendMode(this.p.ADD);
        const glowIntensity = this.brightness * 0.2;
        this.p.fill(r, g, b, glowIntensity);
        this.p.circle(this.x, this.y, size * 2);
        this.p.blendMode(this.p.BLEND);
      }

      if (this.type === StarType.SUPERBRIGHT && this.brightness > 140) {
        this.p.blendMode(this.p.ADD);
        const outerGlow = this.brightness * 0.1;
        this.p.fill(r, g, b, outerGlow);
        this.p.circle(this.x, this.y, size * 3);
        this.p.blendMode(this.p.BLEND);
      }
    }

    private drawStarShape(
      x: number,
      y: number,
      outerRadius: number,
      innerRadius: number,
    ): void {
      const angleStep = this.p.PI / 5;
      this.p.beginShape();
      for (let i = 0; i < 10; i++) {
        const angle = i * angleStep - this.p.PI / 2;
        const radius = i % 2 === 0 ? outerRadius : innerRadius;
        const px = x + this.p.cos(angle) * radius;
        const py = y + this.p.sin(angle) * radius;
        this.p.vertex(px, py);
      }
      this.p.endShape(this.p.CLOSE);
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

    private p: p5;

    constructor(
      p: p5,
      props: Omit<
        Comet,
        "p" | "x" | "y" | "progress" | "active" | "update" | "render"
      >,
    ) {
      this.p = p;
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

    render(): void {
      const [r, g, b] = this.color;

      const dx = this.endX - this.startX;
      const dy = this.endY - this.startY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const tailX = -(dx / distance) * this.tailLength;
      const tailY = -(dy / distance) * this.tailLength;

      const segments = 8;
      for (let i = 0; i < segments; i++) {
        const t = i / segments;
        const alpha = this.brightness * (1 - t) * 0.6;
        const thickness = (1 - t) * 3 + 1;

        this.p.stroke(r, g, b, alpha);
        this.p.strokeWeight(thickness);

        const x1 = this.x + tailX * t;
        const y1 = this.y + tailY * t;
        const x2 = this.x + tailX * (t + 1 / segments);
        const y2 = this.y + tailY * (t + 1 / segments);

        this.p.line(x1, y1, x2, y2);
      }

      this.p.blendMode(this.p.ADD);
      this.p.noStroke();
      this.p.fill(r, g, b, this.brightness);
      this.p.circle(this.x, this.y, 4);

      this.p.fill(255, 255, 255, this.brightness * 0.8);
      this.p.circle(this.x, this.y, 2);
      this.p.blendMode(this.p.BLEND);
    }
  }

  class StarField {
    private p: p5;
    private stars: Star[] = [];
    private comets: Comet[] = [];
    private time = 0;
    private starIdCounter = 0;
    private cometIdCounter = 0;
    private lastCometTime = 0;
    private nextRecalc = 600;
    private staticStarsBuffer: p5.Graphics | null = null;

    constructor(p: p5) {
      this.p = p;
      this.nextRecalc = 600;
      this.generateRealStarField();
    }

    private generateRealStarField(): void {
      this.stars = [];
      this.comets = [];
      this.starIdCounter = 0;
      this.cometIdCounter = 0;

      const now = new Date();
      const jd = 2440587.5 + now.getTime() / 86400000;
      const utc = now.getTime() / 1000 / 3600;
      const lst = ((utc / 24 + NEW_YORK_LON / 360) % 1) * 24;

      for (const starData of BRIGHT_STARS_CATALOG) {
        if (starData.mag > NYC_MAGNITUDE_CUTOFF) {
          continue;
        }

        const hourAngle = (lst * 15 - starData.ra + 360) % 360;

        const sinAlt =
          Math.sin((starData.dec * Math.PI) / 180) *
            Math.sin((NEW_YORK_LAT * Math.PI) / 180) +
          Math.cos((starData.dec * Math.PI) / 180) *
            Math.cos((NEW_YORK_LAT * Math.PI) / 180) *
            Math.cos((hourAngle * Math.PI) / 180);
        const altitude = (Math.asin(sinAlt) * 180) / Math.PI;

        if (altitude < 0) {
          continue;
        }

        const cosA =
          (Math.sin((starData.dec * Math.PI) / 180) -
            Math.sin((altitude * Math.PI) / 180) *
              Math.sin((NEW_YORK_LAT * Math.PI) / 180)) /
          (Math.cos((altitude * Math.PI) / 180) *
            Math.cos((NEW_YORK_LAT * Math.PI) / 180));
        let azimuth =
          (Math.acos(Math.max(-1, Math.min(1, cosA))) * 180) / Math.PI;
        if (Math.sin((hourAngle * Math.PI) / 180) > 0) {
          azimuth = 360 - azimuth;
        }

        const screenCoords = this.projectHorizonToScreen(azimuth, altitude);

        const size = this.mapMagnitudeToSize(starData.mag);
        const maxBrightness = this.mapMagnitudeToBrightness(starData.mag);
        const type = this.determineStarTypeFromMagnitude(starData.mag);
        const color = determineStarColorFromSpectralType(starData.spectralType);

        const isStatic = starData.mag >= 2.0;

        this.stars.push(
          new Star(this.p, {
            id: this.starIdCounter++,
            name: starData.name,
            ra: starData.ra,
            dec: starData.dec,
            mag: starData.mag,
            x: screenCoords.x,
            y: screenCoords.y,
            size,
            maxBrightness,
            flickerSpeed: calculateFlickerSpeed(type),
            noiseOffset: Math.random() * 1000,
            color,
            type,
            shape: getRandomShape(!isStatic),
            isStatic,
          }),
        );
      }

      const minStarsDesired = 350;
      if (this.stars.length < minStarsDesired) {
        const numExtraStars = minStarsDesired - this.stars.length;
        for (let i = 0; i < numExtraStars; i++) {
          const x = Math.random() * this.p.windowWidth;
          const y = Math.random() * this.p.windowHeight;
          this.stars.push(
            new Star(this.p, {
              id: this.starIdCounter++,
              name: "Distant Star",
              ra: 0,
              dec: 0,
              mag: 5.0 + Math.random(),
              x,
              y,
              size: StarSize.TINY,
              maxBrightness: this.mapMagnitudeToBrightness(
                5.5 + Math.random() * 0.5,
              ),
              flickerSpeed: calculateFlickerSpeed(StarType.DISTANT),
              noiseOffset: Math.random() * 1000,
              color: STAR_COLORS[0],
              type: StarType.DISTANT,
              shape: StarShape.CIRCLE,
              isStatic: true,
            }),
          );
        }
      }

      this.stars.sort((a, b) => a.maxBrightness - b.maxBrightness);

      this.renderStaticStarsToBuffer();
    }

    private projectHorizonToScreen(
      azimuthDegrees: number,
      altitudeDegrees: number,
    ): { x: number; y: number } {
      const viewWidth = this.p.windowWidth;
      const viewHeight = this.p.windowHeight;

      let mappedAzimuth = azimuthDegrees;

      const fovY = 90;
      const fovX = 180;
      const y = this.p.map(
        altitudeDegrees,
        0,
        90,
        viewHeight * 0.8,
        viewHeight * 0.1,
      );

      let azDiff = azimuthDegrees - 0;
      if (azDiff > 180) azDiff -= 360;
      if (azDiff < -180) azDiff += 360;

      const visibleAzRange = 120;
      const x = this.p.map(
        azDiff,
        -visibleAzRange / 2,
        visibleAzRange / 2,
        0,
        viewWidth,
      );

      return {
        x: x,
        y: y,
      };
    }

    private mapMagnitudeToSize(mag: number): StarSize {
      if (mag < -0.5) return StarSize.MASSIVE;
      if (mag < 0.5) return StarSize.LARGE;
      if (mag < 1.5) return StarSize.MEDIUM;
      if (mag < 2.5) return StarSize.SMALL;
      return StarSize.TINY;
    }

    private mapMagnitudeToBrightness(mag: number): number {
      const maxMag = NYC_MAGNITUDE_CUTOFF;
      const minMag = -1.5;

      const constrainedMag = this.p.constrain(mag, minMag, maxMag);
      return this.p.map(constrainedMag, maxMag, minMag, 80, 255);
    }

    private determineStarTypeFromMagnitude(mag: number): StarType {
      if (mag < 0) return StarType.SUPERBRIGHT;
      if (mag < 1.0) return StarType.BRIGHT;
      if (mag < 2.5) return StarType.MEDIUM;
      return StarType.DISTANT;
    }

    private renderStaticStarsToBuffer(): void {
      if (this.staticStarsBuffer) {
        this.staticStarsBuffer.remove();
      }
      this.staticStarsBuffer = this.p.createGraphics(
        this.p.windowWidth,
        this.p.windowHeight,
      );

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

      for (const star of this.stars) {
        star.update(this.time, deltaTime);
      }

      for (let i = this.comets.length - 1; i >= 0; i--) {
        const comet = this.comets[i];
        comet.update(deltaTime);
        if (!comet.active) {
          this.comets.splice(i, 1);
        }
      }

      if (this.time >= this.nextRecalc) {
        this.generateRealStarField();
        this.nextRecalc = this.time + 600;
      }

      if (this.time - this.lastCometTime > 15 + Math.random() * 25) {
        this.spawnComet();
        this.lastCometTime = this.time;
      }
    }

    private spawnComet(): void {
      const edge = Math.floor(Math.random() * 4);
      let startX, startY, endX, endY;

      switch (edge) {
        case 0:
          startX = Math.random() * this.p.width * 0.8 + this.p.width * 0.1;
          startY = -50;
          endX = startX + (Math.random() - 0.5) * this.p.width * 0.5;
          endY = this.p.height + 50;
          break;
        case 1:
          startX = this.p.width + 50;
          startY = Math.random() * this.p.height * 0.8 + this.p.height * 0.1;
          endX = -50;
          endY = startY + (Math.random() - 0.5) * this.p.height * 0.5;
          break;
        case 2:
          startX = Math.random() * this.p.width * 0.8 + this.p.width * 0.1;
          startY = this.p.height + 50;
          endX = startX + (Math.random() - 0.5) * this.p.width * 0.5;
          endY = -50;
          break;
        default:
          startX = -50;
          startY = Math.random() * this.p.height * 0.8 + this.p.height * 0.1;
          endX = this.p.width + 50;
          endY = startY + (Math.random() - 0.5) * this.p.height * 0.5;
          break;
      }

      const comet = new Comet(this.p, {
        id: this.cometIdCounter++,
        startX,
        startY,
        endX,
        endY,
        speed: 0.2 + Math.random() * 0.3,
        tailLength: 50 + Math.random() * 80,
        brightness: 180 + Math.random() * 75,
        color: STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)],
      });

      this.comets.push(comet);
    }

    public render(): void {
      this.p.clear();

      if (this.staticStarsBuffer) {
        this.p.image(this.staticStarsBuffer, 0, 0);
      }

      for (const star of this.stars) {
        if (!star.isStatic) {
          star.render();
        }
      }

      for (const comet of this.comets) {
        if (comet.active) {
          comet.render();
        }
      }
    }

    public resize(width: number, height: number): void {
      this.generateRealStarField();
      this.nextRecalc = this.time + 600;
    }

    public getStarCount(): number {
      return this.stars.length;
    }

    public getStarTypes(): Record<StarType, number> {
      return this.stars.reduce(
        (acc, star) => {
          acc[star.type] = (acc[star.type] || 0) + 1;
          return acc;
        },
        {} as Record<StarType, number>,
      );
    }
  }

  onMount(async () => {
    try {
      const p5Module = await import("p5");
      const p5 = p5Module.default;

      const starfieldSketch = (p: p5) => {
        let starField: StarField;
        let lastFrameTime = 0;

        p.setup = () => {
          const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
          canvas.parent(starfieldContainer);
          canvas.style("position", "fixed");
          canvas.style("top", "0");
          canvas.style("left", "0");
          canvas.style("z-index", "-1");
          canvas.style("pointer-events", "none");

          starField = new StarField(p);
          lastFrameTime = p.millis();
        };

        p.draw = () => {
          const currentTime = p.millis();
          const deltaTime = (currentTime - lastFrameTime) / 1000;
          lastFrameTime = currentTime;

          starField.update(deltaTime);
          starField.render();
        };

        p.windowResized = () => {
          p.resizeCanvas(p.windowWidth, p.windowHeight);
          starField.resize(p.windowWidth, p.windowHeight);
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
