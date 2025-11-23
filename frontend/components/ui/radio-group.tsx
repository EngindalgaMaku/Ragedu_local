import * as React from "react";
import { cn } from "@/lib/utils";

const RadioGroupContext = React.createContext<{
  name: string;
  value?: string;
  onValueChange: (value: string) => void;
}>({
  name: "",
  onValueChange: () => {},
});

export interface RadioGroupProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: string;
  onValueChange?: (value: string) => void;
  name?: string;
}

const RadioGroup = React.forwardRef<HTMLDivElement, RadioGroupProps>(
  ({ className, value, onValueChange, name, ...props }, ref) => {
    const groupName =
      name || `radio-group-${Math.random().toString(36).substring(7)}`;

    return (
      <RadioGroupContext.Provider
        value={{
          name: groupName,
          value,
          onValueChange: onValueChange || (() => {}),
        }}
      >
        <div
          className={cn("grid gap-2", className)}
          role="radiogroup"
          ref={ref}
          {...props}
        />
      </RadioGroupContext.Provider>
    );
  }
);
RadioGroup.displayName = "RadioGroup";

export interface RadioGroupItemProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  value: string;
}

const RadioGroupItem = React.forwardRef<HTMLInputElement, RadioGroupItemProps>(
  ({ className, value, ...props }, ref) => {
    const context = React.useContext(RadioGroupContext);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.checked) {
        context.onValueChange(value);
      }
    };

    return (
      <input
        type="radio"
        className={cn(
          "aspect-square h-4 w-4 rounded-full border border-primary text-primary ring-offset-background focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        name={context.name}
        value={value}
        checked={context.value === value}
        onChange={handleChange}
        {...props}
      />
    );
  }
);
RadioGroupItem.displayName = "RadioGroupItem";

export { RadioGroup, RadioGroupItem };
