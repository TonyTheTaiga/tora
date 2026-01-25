export const validators = {
  required: (value: string) =>
    !value?.trim() ? "This field is required" : null,

  minLength:
    (min: number) =>
    (value: string): string | null =>
      value && value.length < min ? `Must be at least ${min} characters` : null,

  maxLength:
    (max: number) =>
    (value: string): string | null =>
      value && value.length > max
        ? `Must be no more than ${max} characters`
        : null,

  email: (value: string): string | null =>
    value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)
      ? "Invalid email address"
      : null,

  noSpaces: (value: string): string | null =>
    value && /\s/.test(value) ? "Cannot contain spaces" : null,

  alphanumeric: (value: string): string | null =>
    value && !/^[a-zA-Z0-9_-]+$/.test(value)
      ? "Only letters, numbers, underscores, and hyphens allowed"
      : null,

  compose:
    (...fns: Array<(v: string) => string | null>) =>
    (value: string): string | null => {
      for (const fn of fns) {
        const error = fn(value);
        if (error) return error;
      }
      return null;
    },
};
