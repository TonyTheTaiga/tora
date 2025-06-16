interface ToolbarButton {
  id: string;
  icon: any;
  onClick: () => void;
  ariaLabel?: string;
  title?: string;
}

let buttons = $state<ToolbarButton[]>([]);

export function addToolbarButton(button: ToolbarButton) {
  buttons = [...buttons, button];
}

export function removeToolbarButton(id: string) {
  buttons = buttons.filter((b) => b.id !== id);
}

export function getToolbarButtons() {
  return buttons;
}
