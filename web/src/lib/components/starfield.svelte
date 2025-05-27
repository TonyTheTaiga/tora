<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import type p5 from "p5"; // Import p5 type for better intellisense

  let starfieldContainer: HTMLDivElement;
  let p5Instance: p5 | null = null;

  // --- Enums and Constants ---
  const StarType = {
    DISTANT: "distant",
    MEDIUM: "medium",
    BRIGHT: "bright",
    SUPERBRIGHT: "superbright",
  } as const;

  type StarType = (typeof StarType)[keyof typeof StarType];

  const StarSize = {
    TINY: 1,
    SMALL: 2,
    MEDIUM: 3,
    LARGE: 4,
    MASSIVE: 5,
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

  // Subtle warm palette with gentle yellow and orange tints
  const STAR_COLORS: readonly [number, number, number][] = [
    [245, 240, 225], // warm white - most stars
    [255, 245, 215], // subtle yellow-white
    [255, 235, 195], // gentle yellow
    [255, 215, 175], // soft orange accent - rare
  ] as const;

  // --- Interfaces (still useful for properties, but now within classes) ---
  interface IStar {
    readonly id: number;
    x: number;
    y: number;
    size: StarSize;
    brightness: number;
    readonly maxBrightness: number;
    readonly flickerSpeed: number;
    readonly flickerOffset: number; // For sine wave
    readonly noiseOffset: number; // For Perlin noise
    readonly color: readonly [number, number, number];
    readonly type: StarType;
    readonly shape: StarShape;
    twinklePhase: number;
    pulsePhase: number;
    isStatic?: boolean; // New: to determine if it should be pre-rendered
  }

  interface IComet {
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
  }

  // --- Helper Functions (moved from StarField class) ---

  // Box-Muller transform for normal distribution
  function normalRandom(): number {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // Easing function for comets
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
    // Much slower, peaceful flicker rates
    switch (type) {
      case StarType.DISTANT:
        return 0.2 + Math.random() * 0.4; // 0.2-0.6 Hz - very slow
      case StarType.MEDIUM:
        return 0.15 + Math.random() * 0.35; // 0.15-0.5 Hz - gentle
      case StarType.BRIGHT:
        return 0.1 + Math.random() * 0.3; // 0.1-0.4 Hz - calm
      case StarType.SUPERBRIGHT:
        return 0.08 + Math.random() * 0.25; // 0.08-0.33 Hz - serene
      default:
        return 0.2;
    }
  }

  // --- Star Class ---
  class Star implements IStar {
    readonly id: number;
    x: number;
    y: number;
    size: StarSize;
    brightness: number;
    readonly maxBrightness: number;
    readonly flickerSpeed: number;
    readonly flickerOffset: number;
    readonly noiseOffset: number;
    readonly color: readonly [number, number, number];
    readonly type: StarType;
    readonly shape: StarShape;
    twinklePhase: number;
    pulsePhase: number;
    isStatic: boolean; // Stars that are static won't update brightness

    private p: p5; // p5 instance passed to each star for drawing

    constructor(
      p: p5,
      props: Omit<IStar, "brightness" | "twinklePhase" | "pulsePhase"> & {
        isStatic: boolean;
      },
    ) {
      this.p = p;
      this.id = props.id;
      this.x = props.x;
      this.y = props.y;
      this.size = props.size;
      this.maxBrightness = props.maxBrightness;
      this.flickerSpeed = props.flickerSpeed;
      this.flickerOffset = props.flickerOffset;
      this.noiseOffset = props.noiseOffset;
      this.color = props.color;
      this.type = props.type;
      this.shape = props.shape;
      this.isStatic = props.isStatic;
      this.brightness = this.maxBrightness * (0.5 + Math.random() * 0.5); // Start at 50-100% brightness
      this.twinklePhase = Math.random() * this.p.TWO_PI;
      this.pulsePhase = Math.random() * this.p.TWO_PI;
    }

    update(time: number, deltaTime: number): void {
      if (this.isStatic) return; // No need to update brightness for static stars

      // Primary gentle Perlin noise flicker for organic feel
      const noiseVal = this.p.noise(
        time * this.flickerSpeed * 0.1 + this.noiseOffset,
      );
      // Map noise (0-1) to a flicker range (-1 to 1)
      const primaryFlicker = this.p.map(noiseVal, 0, 1, -1, 1);

      // Gentle pulse effect for larger stars
      this.pulsePhase += deltaTime * this.flickerSpeed * 0.5;
      const pulseIntensity =
        this.size >= StarSize.MEDIUM ? Math.sin(this.pulsePhase) * 0.15 : 0;

      // Combine effects
      const combinedFlicker = primaryFlicker + pulseIntensity;
      // Normalize combined flicker to a 20-100% range of maxBrightness
      const normalizedFlicker = this.p.constrain(
        (combinedFlicker + 1) * 0.5,
        0.2,
        1,
      );

      // Apply gentle brightness variation for peaceful effect
      let brightnessModifier = 1;
      switch (this.type) {
        case StarType.DISTANT:
          brightnessModifier = 0.6 + normalizedFlicker * 0.4; // 60-100% - stable
          break;
        case StarType.MEDIUM:
          brightnessModifier = 0.5 + normalizedFlicker * 0.5; // 50-100% - gentle
          break;
        case StarType.BRIGHT:
          brightnessModifier = 0.4 + normalizedFlicker * 0.6; // 40-100% - noticeable
          break;
        case StarType.SUPERBRIGHT:
          brightnessModifier = 0.6 + normalizedFlicker * 0.4; // 60-100% - steady beacon
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

      // Main star body with shape variety
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
          // Main cross
          this.p.rect(pixelX - halfSize, pixelY - 0.5, size, 1);
          this.p.rect(pixelX - 0.5, pixelY - halfSize, 1, size);
          // Diagonal crosses
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

      // Add gentle glow for larger stars using additive blending
      if (this.size >= StarSize.MEDIUM && this.brightness > 120) {
        this.p.blendMode(this.p.ADD);
        const glowIntensity = this.brightness * 0.2;
        this.p.fill(r, g, b, glowIntensity);
        this.p.circle(this.x, this.y, size * 2);
        this.p.blendMode(this.p.BLEND); // Revert
      }

      // Add soft outer glow for superbright stars using additive blending
      if (this.type === StarType.SUPERBRIGHT && this.brightness > 140) {
        this.p.blendMode(this.p.ADD);
        const outerGlow = this.brightness * 0.1;
        this.p.fill(r, g, b, outerGlow);
        this.p.circle(this.x, this.y, size * 3);
        this.p.blendMode(this.p.BLEND); // Revert
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

  // --- Comet Class ---
  class Comet implements IComet {
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

    constructor(p: p5, props: Omit<IComet, "x" | "y" | "progress" | "active">) {
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

      // Calculate tail direction
      const dx = this.endX - this.startX;
      const dy = this.endY - this.startY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const tailX = -(dx / distance) * this.tailLength;
      const tailY = -(dy / distance) * this.tailLength;

      // Draw tail with gradient
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

      // Draw comet head with additive blending for extra glow
      this.p.blendMode(this.p.ADD);
      this.p.noStroke();
      this.p.fill(r, g, b, this.brightness);
      this.p.circle(this.x, this.y, 4);

      // Bright center
      this.p.fill(255, 255, 255, this.brightness * 0.8);
      this.p.circle(this.x, this.y, 2);
      this.p.blendMode(this.p.BLEND); // Revert
    }
  }

  // --- StarField Manager Class ---
  class StarField {
    private stars: Star[] = [];
    private comets: Comet[] = [];
    private time = 0;
    private p: p5;
    private starIdCounter = 0;
    private cometIdCounter = 0;
    private lastCometTime = 0;
    private staticStarsBuffer: p5.Graphics | null = null; // New: for pre-rendering static stars

    private readonly TARGET_STARS = 400; // Aim for this many stars
    private readonly MIN_STARS = 150;
    private readonly MAX_STARS = 700;

    constructor(p: p5) {
      this.p = p;
      this.generateStarField();
    }

    private generateStarField(): void {
      this.stars = [];
      this.comets = [];
      this.starIdCounter = 0;
      this.cometIdCounter = 0;

      const screenArea = this.p.windowWidth * this.p.windowHeight;
      // Adjust target stars based on screen size, but within min/max
      const effectiveTargetStars = this.p.constrain(
        Math.floor(this.TARGET_STARS * (screenArea / (1920 * 1080))), // Scale by a common desktop resolution
        this.MIN_STARS,
        this.MAX_STARS,
      );

      console.log(
        `Generating peaceful starfield for ${this.p.windowWidth}x${this.p.windowHeight}. Targeting ${effectiveTargetStars} stars.`,
      );

      // Generate background stars
      const numBackgroundStars = Math.floor(effectiveTargetStars * 0.7); // 70% background
      for (let i = 0; i < numBackgroundStars; i++) {
        this.stars.push(this.createStar(false)); // Not a cluster star
      }

      // Generate fewer, softer star clusters
      this.generateStarClusters(effectiveTargetStars * 0.3); // 30% cluster stars

      // Sort stars by brightness for optimal rendering order (dimmest first)
      this.stars.sort((a, b) => a.maxBrightness - b.maxBrightness);

      // Pre-render static stars to a buffer
      this.renderStaticStarsToBuffer();

      console.log(`Total stars generated: ${this.stars.length}`);
      console.log("Star distribution:", this.getStarTypes());
    }

    private renderStaticStarsToBuffer(): void {
      if (this.staticStarsBuffer) {
        this.staticStarsBuffer.remove(); // Remove previous buffer if exists
      }
      this.staticStarsBuffer = this.p.createGraphics(
        this.p.windowWidth,
        this.p.windowHeight,
      );

      this.staticStarsBuffer.clear(); // Clear the buffer
      this.staticStarsBuffer.noStroke();

      for (const star of this.stars) {
        if (star.isStatic) {
          const [r, g, b] = star.color;
          this.staticStarsBuffer.fill(r, g, b, star.brightness);
          this.staticStarsBuffer.circle(star.x, star.y, star.size); // Static stars are simple circles
        }
      }
      console.log("Static stars pre-rendered to buffer.");
    }

    private generateStarClusters(numClusterStars: number): void {
      const numClusters = Math.floor(Math.random() * 2) + 1; // 1-2 clusters for calmer feel

      for (let cluster = 0; cluster < numClusters; cluster++) {
        const centerX = Math.random() * this.p.windowWidth;
        const centerY = Math.random() * this.p.windowHeight;
        const clusterRadius = 60 + Math.random() * 100; // Smaller, gentler clusters
        const clusterDensity = Math.floor(
          (numClusterStars / numClusters) * (0.8 + Math.random() * 0.4),
        ); // Distribute cluster stars

        console.log(
          `Peaceful cluster ${cluster + 1}: center(${Math.floor(centerX)}, ${Math.floor(centerY)}), radius: ${Math.floor(clusterRadius)}, stars: ${clusterDensity}`,
        );

        for (let i = 0; i < clusterDensity; i++) {
          const angle = Math.random() * this.p.TWO_PI;
          const distance = normalRandom() * clusterRadius * 0.7;

          const x = centerX + Math.cos(angle) * distance;
          const y = centerY + Math.sin(angle) * distance;

          if (
            x >= 0 &&
            x < this.p.windowWidth &&
            y >= 0 &&
            y < this.p.windowHeight
          ) {
            this.stars.push(
              this.createStar(true, x, y, distance, clusterRadius),
            );
          }
        }
      }
    }

    private createStar(
      isClusterStar: boolean,
      x?: number,
      y?: number,
      distanceFromCenter?: number,
      clusterRadius?: number,
    ): Star {
      const starX = x !== undefined ? x : Math.random() * this.p.windowWidth;
      const starY = y !== undefined ? y : Math.random() * this.p.windowHeight;

      const starTypeRoll = Math.random();
      let type: StarType;
      let size: StarSize;
      let maxBrightness: number;
      let colorIndex: number;
      let shape: StarShape;
      let isStatic: boolean;

      if (isClusterStar) {
        const centerProximity = 1 - distanceFromCenter! / clusterRadius!;

        if (starTypeRoll < 0.4) {
          type = StarType.DISTANT;
          size = Math.random() < 0.7 ? StarSize.TINY : StarSize.SMALL;
          maxBrightness = 80 + Math.random() * 60 + centerProximity * 40;
          colorIndex = Math.random() < 0.8 ? 0 : 1;
          shape = Math.random() < 0.7 ? StarShape.CIRCLE : StarShape.DIAMOND;
          isStatic = true; // Distant cluster stars can be static
        } else if (starTypeRoll < 0.7) {
          type = StarType.MEDIUM;
          size = Math.random() < 0.5 ? StarSize.SMALL : StarSize.MEDIUM;
          maxBrightness = 100 + Math.random() * 70 + centerProximity * 50;
          colorIndex = Math.random() < 0.7 ? 0 : Math.random() < 0.7 ? 1 : 2;
          shape = getRandomShape(false);
          isStatic = false;
        } else if (starTypeRoll < 0.9) {
          type = StarType.BRIGHT;
          size = Math.random() < 0.3 ? StarSize.MEDIUM : StarSize.LARGE;
          maxBrightness = 140 + Math.random() * 65 + centerProximity * 30;
          colorIndex = Math.random() < 0.6 ? 0 : Math.random() < 0.5 ? 2 : 3;
          shape = getRandomShape(true);
          isStatic = false;
        } else {
          type = StarType.SUPERBRIGHT;
          size = Math.random() < 0.4 ? StarSize.LARGE : StarSize.MASSIVE;
          maxBrightness = 180 + Math.random() * 45 + centerProximity * 30;
          colorIndex = Math.random() < 0.5 ? 2 : 3;
          shape = Math.random() < 0.5 ? StarShape.STAR : StarShape.TWINKLE;
          isStatic = false;
        }
      } else {
        // Background field stars
        if (starTypeRoll < 0.65) {
          type = StarType.DISTANT;
          size = Math.random() < 0.8 ? StarSize.TINY : StarSize.SMALL;
          maxBrightness = 60 + Math.random() * 50;
          colorIndex = Math.random() < 0.9 ? 0 : 1;
          shape = Math.random() < 0.8 ? StarShape.CIRCLE : StarShape.DIAMOND;
          isStatic = true; // Most distant stars are static
        } else if (starTypeRoll < 0.85) {
          type = StarType.MEDIUM;
          size = Math.random() < 0.6 ? StarSize.SMALL : StarSize.MEDIUM;
          maxBrightness = 90 + Math.random() * 60;
          colorIndex = Math.random() < 0.8 ? 0 : 1;
          shape = getRandomShape(false);
          isStatic = false;
        } else if (starTypeRoll < 0.97) {
          type = StarType.BRIGHT;
          size = Math.random() < 0.4 ? StarSize.MEDIUM : StarSize.LARGE;
          maxBrightness = 120 + Math.random() * 55;
          colorIndex = Math.random() < 0.7 ? 0 : 2;
          shape = getRandomShape(true);
          isStatic = false;
        } else {
          type = StarType.SUPERBRIGHT;
          size = Math.random() < 0.5 ? StarSize.LARGE : StarSize.MASSIVE;
          maxBrightness = 160 + Math.random() * 40;
          colorIndex = Math.random() < 0.4 ? 2 : 3;
          shape = Math.random() < 0.6 ? StarShape.STAR : StarShape.TWINKLE;
          isStatic = false;
        }
      }

      return new Star(this.p, {
        id: this.starIdCounter++,
        x: starX,
        y: starY,
        size,
        maxBrightness,
        flickerSpeed: calculateFlickerSpeed(type),
        flickerOffset: Math.random() * this.p.TWO_PI,
        noiseOffset: Math.random() * 1000, // Unique noise offset for each star
        color: STAR_COLORS[colorIndex],
        type,
        shape,
        isStatic, // Assign the static property
      });
    }

    public update(deltaTime: number): void {
      this.time += deltaTime;

      // Update non-static stars
      for (const star of this.stars) {
        star.update(this.time, deltaTime);
      }

      // Update comets
      for (let i = this.comets.length - 1; i >= 0; i--) {
        const comet = this.comets[i];
        comet.update(deltaTime);
        if (!comet.active) {
          this.comets.splice(i, 1);
        }
      }

      // Occasionally spawn new comets (less frequent, peaceful)
      if (this.time - this.lastCometTime > 15 + Math.random() * 25) {
        // Every 15-40 seconds
        this.spawnComet();
        this.lastCometTime = this.time;
      }
    }

    private spawnComet(): void {
      const edge = Math.floor(Math.random() * 4); // 0=top, 1=right, 2=bottom, 3=left
      let startX, startY, endX, endY;

      // Define entry and exit points for more controlled trajectories
      switch (edge) {
        case 0: // Top edge to bottom-right or bottom-left
          startX = Math.random() * this.p.width * 0.8 + this.p.width * 0.1; // Start in middle 80% of top
          startY = -50;
          endX = Math.random() < 0.5 ? this.p.width + 50 : -50; // Either off right or off left
          endY = this.p.height + 50;
          break;
        case 1: // Right edge to bottom-left or top-left
          startX = this.p.width + 50;
          startY = Math.random() * this.p.height * 0.8 + this.p.height * 0.1;
          endX = -50;
          endY = Math.random() < 0.5 ? this.p.height + 50 : -50;
          break;
        case 2: // Bottom edge to top-right or top-left
          startX = Math.random() * this.p.width * 0.8 + this.p.width * 0.1;
          startY = this.p.height + 50;
          endX = Math.random() < 0.5 ? this.p.width + 50 : -50;
          endY = -50;
          break;
        default: // Left edge to top-right or bottom-right
          startX = -50;
          startY = Math.random() * this.p.height * 0.8 + this.p.height * 0.1;
          endX = this.p.width + 50;
          endY = Math.random() < 0.5 ? this.p.height + 50 : -50;
          break;
      }

      const comet = new Comet(this.p, {
        id: this.cometIdCounter++,
        startX,
        startY,
        endX,
        endY,
        speed: 0.2 + Math.random() * 0.3, // Slower: 0.2-0.5
        tailLength: 50 + Math.random() * 80, // 50-130 pixel tail
        brightness: 180 + Math.random() * 75, // 180-255
        color: STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)], // Random from all defined colors
      });

      this.comets.push(comet);
      console.log(`Peaceful comet spawned: ${comet.id}`);
    }

    public render(): void {
      this.p.clear(); // Clear canvas each frame

      // Draw the pre-rendered static star layer first
      if (this.staticStarsBuffer) {
        this.p.image(this.staticStarsBuffer, 0, 0);
      }

      // Render non-static stars (which update every frame)
      for (const star of this.stars) {
        if (!star.isStatic) {
          star.render();
        }
      }

      // Render comets on top
      for (const comet of this.comets) {
        if (comet.active) {
          comet.render();
        }
      }
    }

    public resize(width: number, height: number): void {
      console.log(`Resizing starfield to ${width}x${height}`);
      this.generateStarField(); // Re-generate stars and buffer on resize
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
      // Dynamic import of p5.js
      const p5Module = await import("p5");
      const p5 = p5Module.default;
      console.log("p5.js loaded for starfield");

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
          // deltaTime in seconds for consistent frame rate independence
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
  /* Ensure the container fills the viewport */
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
