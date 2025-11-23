import * as React from "react";
import { cn } from "@/lib/utils";

interface DialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: React.ReactNode;
}

interface DialogContentProps {
  className?: string;
  children: React.ReactNode;
}

interface DialogHeaderProps {
  className?: string;
  children: React.ReactNode;
}

interface DialogTitleProps {
  className?: string;
  children: React.ReactNode;
}

interface DialogDescriptionProps {
  className?: string;
  children: React.ReactNode;
}

interface DialogFooterProps {
  className?: string;
  children: React.ReactNode;
}

const Dialog: React.FC<DialogProps> = ({ open, onOpenChange, children }) => {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="fixed inset-0 bg-black/50"
        onClick={() => onOpenChange?.(false)}
      />
      <div className="relative z-50">{children}</div>
    </div>
  );
};

const DialogContent: React.FC<DialogContentProps> = ({
  className,
  children,
}) => (
  <div
    className={cn(
      "bg-white rounded-lg shadow-xl p-6 w-full max-w-md mx-4",
      className
    )}
  >
    {children}
  </div>
);

const DialogHeader: React.FC<DialogHeaderProps> = ({ className, children }) => (
  <div className={cn("mb-4", className)}>{children}</div>
);

const DialogTitle: React.FC<DialogTitleProps> = ({ className, children }) => (
  <h2 className={cn("text-lg font-semibold", className)}>{children}</h2>
);

const DialogDescription: React.FC<DialogDescriptionProps> = ({
  className,
  children,
}) => <p className={cn("text-sm text-gray-600 mt-2", className)}>{children}</p>;

const DialogFooter: React.FC<DialogFooterProps> = ({ className, children }) => (
  <div className={cn("flex justify-end space-x-2 mt-6", className)}>
    {children}
  </div>
);

const DialogTrigger: React.FC<{
  children: React.ReactNode;
  asChild?: boolean;
}> = ({ children }) => <>{children}</>;

export {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogTrigger,
};
