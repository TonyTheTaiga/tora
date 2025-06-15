- prefer svelte5 attachments over $effects
- don't leave comments
- run pnpm run check and pnpm exec tsc --noEmit to verify your work
- always use the styles defined in ./src/app.css
- always clean up unnecessary css when making changes

## Functional Utilitarian UI Design Language

This application follows a terminal/CLI-inspired, functional utilitarian design approach:

### Design Principles

- **Minimal glassmorphism**: Subtle transparency effects with `bg-ctp-surface0/10`, `backdrop-blur-md`
- **Terminal aesthetics**: Monospace fonts (`font-mono`), CLI-style prompts and indicators
- **Information density**: Maximize useful content, minimize decorative elements
- **Functional navigation**: Use brackets for toggles like `[chart]` `[data]`, `[show more]`
- **Lowercase language**: Use lowercase for actions and labels ("back to workspace", "select metrics")

### Key Patterns

- **File listing style**: Headers with bullet points, indented metadata, hierarchical information
- **Metric classification**: Separate scalar (1D) and time series (XD) data with clear navigation
- **Chart/data toggle**: Consistent `[chart]` and `[data]` navigation patterns across components
- **Color semantics**: Green for active/running, red for private/failed, blue for interactive elements
- **Responsive layout**: Mobile-first with condensed information hierarchy

### Implementation Notes

- Metrics should be grouped by name first, then classified by data point count (1 = scalar, >1 = time series)
- Show more/less functionality should use dynamic heights, not fixed scrollable containers
- Navigation between different data views should follow the established bracket notation pattern
- Maintain consistent spacing and visual hierarchy with the terminal-inspired aesthetic
