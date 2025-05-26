<script lang="ts">
  import { onMount, onDestroy } from "svelte";

  let starfieldContainer: HTMLDivElement;
  let p5Instance: import('p5') | null = null;

  interface Star {
    readonly id: number;
    x: number;
    y: number;
    size: StarSize;
    brightness: number;
    readonly maxBrightness: number;
    readonly flickerSpeed: number;
    readonly flickerOffset: number;
    readonly color: readonly [number, number, number];
    readonly type: StarType;
    readonly shape: StarShape;
    lastFlickerTime: number;
    flickerDirection: number;
    twinklePhase: number;
    pulsePhase: number;
  }

  interface Comet {
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

  const StarType = {
    DISTANT: 'distant',
    MEDIUM: 'medium', 
    BRIGHT: 'bright',
    SUPERBRIGHT: 'superbright'
  } as const;
  
  type StarType = typeof StarType[keyof typeof StarType];

  const StarSize = {
    TINY: 1,
    SMALL: 2,
    MEDIUM: 3,
    LARGE: 4,
    MASSIVE: 5
  } as const;
  
  type StarSize = typeof StarSize[keyof typeof StarSize];

  const StarShape = {
    CIRCLE: 'circle',
    DIAMOND: 'diamond',
    PLUS: 'plus',
    STAR: 'star',
    TWINKLE: 'twinkle'
  } as const;
  
  type StarShape = typeof StarShape[keyof typeof StarShape];

  class StarField {
    private stars: Star[] = [];
    private comets: Comet[] = [];
    private time = 0;
    private p: import('p5');
    private starIdCounter = 0;
    private cometIdCounter = 0;
    private lastCometTime = 0;

    // Subtle warm palette with gentle yellow and orange tints
    private readonly STAR_COLORS: readonly [number, number, number][] = [
      [245, 240, 225], // warm white - most stars
      [255, 245, 215], // subtle yellow-white
      [255, 235, 195], // gentle yellow
      [255, 215, 175], // soft orange accent - rare
    ] as const;

    constructor(p: import('p5')) {
      this.p = p;
      this.generateStarField();
    }

    private generateStarField(): void {
      this.stars = [];
      this.comets = [];
      this.starIdCounter = 0;
      this.cometIdCounter = 0;
      
      const screenArea = this.p.windowWidth * this.p.windowHeight;
      const baseDensity = Math.floor(screenArea / 4000); // Reduced for chill vibe
      
      console.log(`Generating peaceful starfield for ${this.p.windowWidth}x${this.p.windowHeight}`);

      // Generate background stars
      for (let i = 0; i < baseDensity; i++) {
        this.stars.push(this.createStar());
      }

      // Generate fewer, softer star clusters
      this.generateStarClusters();

      // Sort stars by brightness for optimal rendering order
      this.stars.sort((a, b) => a.maxBrightness - b.maxBrightness);
      
      console.log(`Total stars generated: ${this.stars.length}`);
    }

    private generateStarClusters(): void {
      const numClusters = Math.floor(Math.random() * 3) + 1; // 1-3 clusters for calmer feel
      
      for (let cluster = 0; cluster < numClusters; cluster++) {
        const centerX = Math.random() * this.p.windowWidth;
        const centerY = Math.random() * this.p.windowHeight;
        const clusterRadius = 60 + Math.random() * 100; // Smaller, gentler clusters
        const clusterDensity = Math.floor(8 + Math.random() * 15); // 8-23 stars per cluster
        
        console.log(`Peaceful cluster ${cluster + 1}: center(${Math.floor(centerX)}, ${Math.floor(centerY)}), radius: ${Math.floor(clusterRadius)}, stars: ${clusterDensity}`);
        
        for (let i = 0; i < clusterDensity; i++) {
          // Use normal distribution for cluster positioning
          const angle = Math.random() * Math.PI * 2;
          const distance = this.normalRandom() * clusterRadius * 0.7; // Slightly more spread
          
          const x = centerX + Math.cos(angle) * distance;
          const y = centerY + Math.sin(angle) * distance;
          
          // Keep stars within screen bounds
          if (x >= 0 && x < this.p.windowWidth && y >= 0 && y < this.p.windowHeight) {
            this.stars.push(this.createClusterStar(x, y, distance, clusterRadius));
          }
        }
      }
    }

    private normalRandom(): number {
      // Box-Muller transform for normal distribution
      let u = 0, v = 0;
      while(u === 0) u = Math.random(); // Converting [0,1) to (0,1)
      while(v === 0) v = Math.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    private createStar(): Star {
      return this.createStarAt(
        Math.random() * this.p.windowWidth,
        Math.random() * this.p.windowHeight,
        false
      );
    }

    private createClusterStar(x: number, y: number, distanceFromCenter: number, clusterRadius: number): Star {
      return this.createStarAt(x, y, true, distanceFromCenter, clusterRadius);
    }

    private createStarAt(x: number, y: number, isClusterStar: boolean, distanceFromCenter?: number, clusterRadius?: number): Star {
      const starTypeRoll = Math.random();
      let type: StarType;
      let size: StarSize;
      let maxBrightness: number;
      let colorIndex: number;
      let shape: StarShape;

      if (isClusterStar) {
        // Cluster stars are generally brighter and more varied
        const centerProximity = 1 - (distanceFromCenter! / clusterRadius!);
        
        if (starTypeRoll < 0.4) {
          type = StarType.DISTANT;
          size = Math.random() < 0.7 ? StarSize.TINY : StarSize.SMALL;
          maxBrightness = 80 + Math.random() * 60 + (centerProximity * 40); // Gentler brightness
          colorIndex = Math.random() < 0.8 ? 0 : 1; // Mostly warm white
          shape = Math.random() < 0.7 ? StarShape.CIRCLE : StarShape.DIAMOND;
        } else if (starTypeRoll < 0.7) {
          type = StarType.MEDIUM;
          size = Math.random() < 0.5 ? StarSize.SMALL : StarSize.MEDIUM;
          maxBrightness = 100 + Math.random() * 70 + (centerProximity * 50);
          colorIndex = Math.random() < 0.7 ? 0 : (Math.random() < 0.7 ? 1 : 2);
          shape = this.getRandomShape(false);
        } else if (starTypeRoll < 0.9) {
          type = StarType.BRIGHT;
          size = Math.random() < 0.3 ? StarSize.MEDIUM : StarSize.LARGE;
          maxBrightness = 140 + Math.random() * 65 + (centerProximity * 30);
          colorIndex = Math.random() < 0.6 ? 0 : (Math.random() < 0.5 ? 2 : 3);
          shape = this.getRandomShape(true);
        } else {
          type = StarType.SUPERBRIGHT;
          size = Math.random() < 0.4 ? StarSize.LARGE : StarSize.MASSIVE;
          maxBrightness = 180 + Math.random() * 45 + (centerProximity * 30);
          colorIndex = Math.random() < 0.5 ? 2 : 3; // Gentle yellow/orange for brightest
          shape = Math.random() < 0.5 ? StarShape.STAR : StarShape.TWINKLE;
        }
      } else {
        // Background field stars - gentler and more peaceful
        if (starTypeRoll < 0.65) {
          type = StarType.DISTANT;
          size = Math.random() < 0.8 ? StarSize.TINY : StarSize.SMALL;
          maxBrightness = 60 + Math.random() * 50; // Softer max brightness
          colorIndex = Math.random() < 0.9 ? 0 : 1; // Almost all warm white
          shape = Math.random() < 0.8 ? StarShape.CIRCLE : StarShape.DIAMOND;
        } else if (starTypeRoll < 0.85) {
          type = StarType.MEDIUM;
          size = Math.random() < 0.6 ? StarSize.SMALL : StarSize.MEDIUM;
          maxBrightness = 90 + Math.random() * 60;
          colorIndex = Math.random() < 0.8 ? 0 : 1;
          shape = this.getRandomShape(false);
        } else if (starTypeRoll < 0.97) {
          type = StarType.BRIGHT;
          size = Math.random() < 0.4 ? StarSize.MEDIUM : StarSize.LARGE;
          maxBrightness = 120 + Math.random() * 55;
          colorIndex = Math.random() < 0.7 ? 0 : 2; // Occasional gentle yellow
          shape = this.getRandomShape(true);
        } else {
          type = StarType.SUPERBRIGHT;
          size = Math.random() < 0.5 ? StarSize.LARGE : StarSize.MASSIVE;
          maxBrightness = 160 + Math.random() * 40;
          colorIndex = Math.random() < 0.4 ? 2 : 3; // Rare warm orange accents
          shape = Math.random() < 0.6 ? StarShape.STAR : StarShape.TWINKLE;
        }
      }

      return {
        id: this.starIdCounter++,
        x,
        y,
        size,
        brightness: maxBrightness * (0.5 + Math.random() * 0.5), // Start at 50-100% brightness
        maxBrightness,
        flickerSpeed: this.calculateFlickerSpeed(type),
        flickerOffset: Math.random() * Math.PI * 2,
        color: this.STAR_COLORS[colorIndex],
        type,
        shape,
        lastFlickerTime: 0,
        flickerDirection: Math.random() > 0.5 ? 1 : -1,
        twinklePhase: Math.random() * Math.PI * 2,
        pulsePhase: Math.random() * Math.PI * 2
      };
    }

    private getRandomShape(isBright: boolean): StarShape {
      if (isBright) {
        const shapes = [StarShape.CIRCLE, StarShape.DIAMOND, StarShape.PLUS, StarShape.STAR, StarShape.TWINKLE];
        return shapes[Math.floor(Math.random() * shapes.length)];
      } else {
        const shapes = [StarShape.CIRCLE, StarShape.DIAMOND, StarShape.PLUS];
        return shapes[Math.floor(Math.random() * shapes.length)];
      }
    }

    private calculateFlickerSpeed(type: StarType): number {
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

    public update(deltaTime: number): void {
      this.time += deltaTime;

      // Update stars
      for (const star of this.stars) {
        this.updateStarBrightness(star, deltaTime);
      }

      // Update comets
      this.updateComets(deltaTime);

      // Occasionally spawn new comets
      if (this.time - this.lastCometTime > 10 + Math.random() * 20) { // Every 10-30 seconds
        this.spawnComet();
        this.lastCometTime = this.time;
      }
    }

    private spawnComet(): void {
      const side = Math.floor(Math.random() * 4); // 0=top, 1=right, 2=bottom, 3=left
      let startX, startY, endX, endY;

      // Start from random edge, travel across screen
      switch (side) {
        case 0: // Top to bottom-right
          startX = Math.random() * this.p.windowWidth;
          startY = -50;
          endX = startX + (Math.random() - 0.5) * this.p.windowWidth;
          endY = this.p.windowHeight + 50;
          break;
        case 1: // Right to bottom-left
          startX = this.p.windowWidth + 50;
          startY = Math.random() * this.p.windowHeight;
          endX = -50;
          endY = startY + (Math.random() - 0.5) * this.p.windowHeight;
          break;
        case 2: // Bottom to top-left
          startX = Math.random() * this.p.windowWidth;
          startY = this.p.windowHeight + 50;
          endX = startX + (Math.random() - 0.5) * this.p.windowWidth;
          endY = -50;
          break;
        default: // Left to top-right
          startX = -50;
          startY = Math.random() * this.p.windowHeight;
          endX = this.p.windowWidth + 50;
          endY = startY + (Math.random() - 0.5) * this.p.windowHeight;
          break;
      }

      const comet: Comet = {
        id: this.cometIdCounter++,
        x: startX,
        y: startY,
        startX,
        startY,
        endX,
        endY,
        speed: 0.3 + Math.random() * 0.4, // 0.3-0.7 - slow, peaceful movement
        tailLength: 40 + Math.random() * 60, // 40-100 pixel tail
        brightness: 150 + Math.random() * 105, // 150-255
        color: this.STAR_COLORS[Math.random() < 0.6 ? 0 : (Math.random() < 0.7 ? 2 : 3)], // Mostly warm white, occasional yellow/orange
        progress: 0,
        active: true
      };

      this.comets.push(comet);
      console.log(`Peaceful comet spawned: ${comet.id}`);
    }

    private updateComets(deltaTime: number): void {
      for (let i = this.comets.length - 1; i >= 0; i--) {
        const comet = this.comets[i];
        
        if (!comet.active) {
          this.comets.splice(i, 1);
          continue;
        }

        // Update position along the path
        comet.progress += deltaTime * comet.speed;
        
        if (comet.progress >= 1) {
          comet.active = false;
          continue;
        }

        // Smooth easing for natural movement
        const easedProgress = this.easeInOutQuad(comet.progress);
        comet.x = comet.startX + (comet.endX - comet.startX) * easedProgress;
        comet.y = comet.startY + (comet.endY - comet.startY) * easedProgress;
      }
    }

    private easeInOutQuad(t: number): number {
      return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    }

    private updateStarBrightness(star: Star, deltaTime: number): void {
      // Individual timing for each star
      star.lastFlickerTime += deltaTime;
      
      // Primary gentle sine wave flicker
      const primaryFlicker = Math.sin(this.time * star.flickerSpeed * 2 * Math.PI + star.flickerOffset);
      
      // Gentle pulse effect for larger stars
      star.pulsePhase += deltaTime * star.flickerSpeed * 0.5;
      const pulseIntensity = star.size >= StarSize.MEDIUM ? 
        Math.sin(star.pulsePhase) * 0.15 : 0;
      
      // Very subtle random variation for organic feel
      const randomFlicker = (Math.random() - 0.5) * 0.05;
      
      // Combine effects with gentle blending
      const combinedFlicker = primaryFlicker + pulseIntensity + randomFlicker;
      const normalizedFlicker = Math.max(0.2, Math.min(1, (combinedFlicker + 1) * 0.5)); // 20-100% range
      
      // Apply gentle brightness variation for peaceful effect
      let brightnessModifier = 1;
      switch (star.type) {
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
      
      star.brightness = star.maxBrightness * brightnessModifier;
    }

    public render(): void {
      this.p.clear();

      // Render stars from dimmest to brightest for proper layering
      for (const star of this.stars) {
        this.renderStar(star);
      }

      // Render comets on top
      for (const comet of this.comets) {
        if (comet.active) {
          this.renderComet(comet);
        }
      }
    }

    private renderStar(star: Star): void {
      const [r, g, b] = star.color;
      const pixelX = Math.floor(star.x);
      const pixelY = Math.floor(star.y);
      const size = star.size;
      const halfSize = Math.floor(size / 2);

      // Main star body with shape variety
      this.p.fill(r, g, b, star.brightness);
      this.p.noStroke();

      switch (star.shape) {
        case StarShape.CIRCLE:
          this.p.circle(star.x, star.y, size);
          break;
          
        case StarShape.DIAMOND:
          this.p.beginShape();
          this.p.vertex(star.x, star.y - halfSize);
          this.p.vertex(star.x + halfSize, star.y);
          this.p.vertex(star.x, star.y + halfSize);
          this.p.vertex(star.x - halfSize, star.y);
          this.p.endShape(this.p.CLOSE);
          break;
          
        case StarShape.PLUS:
          // Horizontal bar
          this.p.rect(pixelX - halfSize, pixelY - 0.5, size, 1);
          // Vertical bar
          this.p.rect(pixelX - 0.5, pixelY - halfSize, 1, size);
          break;
          
        case StarShape.STAR:
          this.drawStarShape(star.x, star.y, size * 0.6, size * 0.3);
          break;
          
        case StarShape.TWINKLE:
          // Main cross
          this.p.rect(pixelX - halfSize, pixelY - 0.5, size, 1);
          this.p.rect(pixelX - 0.5, pixelY - halfSize, 1, size);
          // Diagonal crosses
          const diagonalLength = size * 0.7;
          this.p.stroke(r, g, b, star.brightness);
          this.p.strokeWeight(1);
          this.p.line(star.x - diagonalLength/2, star.y - diagonalLength/2, 
                     star.x + diagonalLength/2, star.y + diagonalLength/2);
          this.p.line(star.x - diagonalLength/2, star.y + diagonalLength/2, 
                     star.x + diagonalLength/2, star.y - diagonalLength/2);
          this.p.noStroke();
          break;
      }

      // Add gentle glow for larger stars
      if (star.size >= StarSize.MEDIUM && star.brightness > 120) {
        const glowIntensity = star.brightness * 0.2;
        this.p.fill(r, g, b, glowIntensity);
        this.p.circle(star.x, star.y, size * 2);
      }

      // Add soft outer glow for superbright stars
      if (star.type === StarType.SUPERBRIGHT && star.brightness > 140) {
        const outerGlow = star.brightness * 0.1;
        this.p.fill(r, g, b, outerGlow);
        this.p.circle(star.x, star.y, size * 3);
      }
    }

    private drawStarShape(x: number, y: number, outerRadius: number, innerRadius: number): void {
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

    private renderComet(comet: Comet): void {
      const [r, g, b] = comet.color;
      
      // Calculate tail direction
      const dx = comet.endX - comet.startX;
      const dy = comet.endY - comet.startY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const tailX = -(dx / distance) * comet.tailLength;
      const tailY = -(dy / distance) * comet.tailLength;

      // Draw tail with gradient
      const segments = 8;
      for (let i = 0; i < segments; i++) {
        const t = i / segments;
        const alpha = comet.brightness * (1 - t) * 0.6;
        const thickness = (1 - t) * 3 + 1;
        
        this.p.stroke(r, g, b, alpha);
        this.p.strokeWeight(thickness);
        
        const x1 = comet.x + tailX * t;
        const y1 = comet.y + tailY * t;
        const x2 = comet.x + tailX * (t + 1/segments);
        const y2 = comet.y + tailY * (t + 1/segments);
        
        this.p.line(x1, y1, x2, y2);
      }

      // Draw comet head
      this.p.noStroke();
      this.p.fill(r, g, b, comet.brightness);
      this.p.circle(comet.x, comet.y, 4);
      
      // Bright center
      this.p.fill(255, 255, 255, comet.brightness * 0.8);
      this.p.circle(comet.x, comet.y, 2);
    }

    public resize(width: number, height: number): void {
      console.log(`Resizing starfield to ${width}x${height}`);
      this.generateStarField();
    }

    public getStarCount(): number {
      return this.stars.length;
    }

    public getStarTypes(): Record<StarType, number> {
      return this.stars.reduce((acc, star) => {
        acc[star.type] = (acc[star.type] || 0) + 1;
        return acc;
      }, {} as Record<StarType, number>);
    }
  }

  onMount(async () => {
    try {
      const p5 = (await import('p5')).default;
      console.log('p5.js loaded for starfield');
      
      const starfieldSketch = (p: import('p5')) => {
        let starField: StarField;
        let lastFrameTime = 0;

        p.setup = () => {
          const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
          canvas.parent(starfieldContainer);
          canvas.style('position', 'fixed');
          canvas.style('top', '0');
          canvas.style('left', '0');
          canvas.style('z-index', '-1');
          canvas.style('pointer-events', 'none');
          
          starField = new StarField(p);
          lastFrameTime = p.millis();
          
          console.log(`Starfield initialized: ${starField.getStarCount()} stars`);
          console.log('Star distribution:', starField.getStarTypes());
        };

        p.draw = () => {
          const currentTime = p.millis();
          const deltaTime = (currentTime - lastFrameTime) / 1000; // Convert to seconds
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
      console.error('Failed to load starfield:', error);
    }
  });

  onDestroy(() => {
    if (p5Instance) {
      p5Instance.remove();
      p5Instance = null;
    }
  });
</script>

<div bind:this={starfieldContainer} class="fixed inset-0 pointer-events-none"></div>