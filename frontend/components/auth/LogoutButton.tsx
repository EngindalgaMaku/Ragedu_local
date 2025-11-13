/**
 * Enhanced LogoutButton Component for RAG Education Assistant
 * Provides various logout options with confirmation and loading states
 */

"use client";

import React, { useState } from "react";
import { LogoutButtonProps } from "../../types/auth";
import { useLogout, useRoles } from "../../hooks/useAuth";

// ===== CONFIRMATION MODAL =====

interface ConfirmationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText: string;
  isLoading: boolean;
}

const ConfirmationModal: React.FC<ConfirmationModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText,
  isLoading,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm">
      <div className="bg-card rounded-2xl shadow-2xl p-6 w-full max-w-md border border-border">
        <div className="flex items-center mb-4">
          <div className="w-10 h-10 bg-destructive/10 rounded-full flex items-center justify-center mr-3">
            <svg
              className="w-5 h-5 text-destructive"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
              />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-foreground">{title}</h3>
        </div>

        <p className="text-muted-foreground mb-6">{message}</p>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            disabled={isLoading}
            className="flex-1 px-4 py-2 text-foreground bg-background border border-border rounded-lg hover:bg-muted transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={isLoading}
            className="flex-1 px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-destructive-foreground/50" />
                <span>Signing out...</span>
              </>
            ) : (
              confirmText
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// ===== MAIN LOGOUT BUTTON COMPONENT =====

const LogoutButton: React.FC<LogoutButtonProps> = ({
  onLogout,
  logoutAll = false,
  className = "",
  children,
}) => {
  const { logout, logoutAll: logoutAllSessions, isLoading } = useLogout();
  const { userRole, isAdmin } = useRoles();
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [isLogoutAll, setIsLogoutAll] = useState(logoutAll);

  // ===== HANDLERS =====

  const handleLogoutClick = () => {
    if (logoutAll || isAdmin) {
      // Show confirmation for admin or when logging out all sessions
      setShowConfirmation(true);
    } else {
      // Direct logout for regular users, single session
      handleConfirmLogout();
    }
  };

  const handleConfirmLogout = async () => {
    try {
      if (logoutAll) {
        await logoutAllSessions();
      } else {
        await logout();
      }

      if (onLogout) {
        onLogout();
      }
    } catch (error) {
      console.error("Logout failed:", error);
    } finally {
      setShowConfirmation(false);
      setShowOptions(false);
    }
  };

  const handleOptionsToggle = () => {
    setShowOptions(!showOptions);
  };

  // ===== RENDER HELPERS =====

  const renderDefaultButton = () => (
    <button
      onClick={handleLogoutClick}
      disabled={isLoading}
      className={`
        inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg
        transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed
        ${
          className ||
          "text-destructive hover:bg-destructive/10 focus:bg-destructive/10"
        }
      `}
      aria-label="Sign out"
    >
      <svg
        className="w-4 h-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
        />
      </svg>
      {isLoading ? "Signing out..." : "Sign Out"}
    </button>
  );

  const renderOptionsMenu = () => {
    if (!showOptions) return null;

    return (
      <div className="absolute right-0 mt-2 w-48 bg-card rounded-lg shadow-lg border border-border z-10">
        <div className="py-1">
          <button
            onClick={() => {
              setIsLogoutAll(false);
              handleLogoutClick();
            }}
            disabled={isLoading}
            className="w-full px-4 py-2 text-left text-sm text-foreground hover:bg-muted flex items-center gap-2"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            Sign out current session
          </button>

          <button
            onClick={() => {
              setIsLogoutAll(true);
              handleLogoutClick();
            }}
            disabled={isLoading}
            className="w-full px-4 py-2 text-left text-sm text-destructive hover:bg-destructive/10 flex items-center gap-2"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.664-.833-2.464 0L4.35 16.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
            Sign out all sessions
          </button>
        </div>
      </div>
    );
  };

  // ===== MAIN RENDER =====

  return (
    <>
      <div className="relative">
        {children ? (
          <div onClick={handleLogoutClick} className={className}>
            {children}
          </div>
        ) : isAdmin ? (
          <div className="relative">
            <button
              onClick={handleOptionsToggle}
              disabled={isLoading}
              className={`
                inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg
                transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed
                ${
                  className ||
                  "text-destructive hover:bg-destructive/10 focus:bg-destructive/10"
                }
              `}
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                />
              </svg>
              {isLoading ? "Signing out..." : "Sign Out"}
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>
            {renderOptionsMenu()}
          </div>
        ) : (
          renderDefaultButton()
        )}
      </div>

      <ConfirmationModal
        isOpen={showConfirmation}
        onClose={() => setShowConfirmation(false)}
        onConfirm={handleConfirmLogout}
        title={logoutAll ? "Sign Out All Sessions" : "Confirm Sign Out"}
        message={
          logoutAll
            ? "This will sign you out from all devices and browsers. Are you sure?"
            : "Are you sure you want to sign out?"
        }
        confirmText={logoutAll ? "Sign Out All" : "Sign Out"}
        isLoading={isLoading}
      />
    </>
  );
};

// ===== SPECIALIZED LOGOUT BUTTON VARIANTS =====

/**
 * Simple logout button without confirmation
 */
export const QuickLogoutButton: React.FC<
  Pick<LogoutButtonProps, "onLogout" | "className">
> = (props) => <LogoutButton {...props} />;

/**
 * Logout button with all sessions option
 */
export const LogoutAllButton: React.FC<
  Pick<LogoutButtonProps, "onLogout" | "className">
> = (props) => <LogoutButton {...props} logoutAll />;

/**
 * Icon-only logout button for compact spaces
 */
export const LogoutIconButton: React.FC<
  Pick<LogoutButtonProps, "onLogout" | "className">
> = ({ onLogout, className = "" }) => (
  <LogoutButton onLogout={onLogout} className={className}>
    <button
      className="p-2 rounded-lg text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
      aria-label="Sign out"
      title="Sign out"
    >
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
        />
      </svg>
    </button>
  </LogoutButton>
);

/**
 * Menu item logout button for dropdowns
 */
export const LogoutMenuItem: React.FC<Pick<LogoutButtonProps, "onLogout">> = ({
  onLogout,
}) => (
  <LogoutButton
    onLogout={onLogout}
    className="w-full px-4 py-2 text-left text-sm text-destructive hover:bg-destructive/10 flex items-center gap-2"
  />
);

/**
 * Danger button style logout
 */
export const DangerLogoutButton: React.FC<
  Pick<LogoutButtonProps, "onLogout" | "className">
> = ({ onLogout, className = "" }) => (
  <LogoutButton
    onLogout={onLogout}
    className={`bg-destructive text-destructive-foreground hover:bg-destructive/90 px-4 py-2 rounded-lg font-medium transition-colors ${className}`}
  />
);

// ===== HIGHER ORDER COMPONENT =====

/**
 * HOC to add logout functionality to any component
 */
export function withLogout<P extends object>(
  Component: React.ComponentType<P>
) {
  const WrappedComponent = (props: P & { onLogout?: () => void }) => {
    const { logout } = useLogout();

    const handleLogout = async () => {
      await logout();
      if (props.onLogout) {
        props.onLogout();
      }
    };

    return <Component {...props} onLogout={handleLogout} />;
  };

  WrappedComponent.displayName = `withLogout(${
    Component.displayName || Component.name
  })`;

  return WrappedComponent;
}

// ===== UTILITY FUNCTIONS =====

/**
 * Create a custom logout button with specific behavior
 */
export function createLogoutButton(defaultProps: Partial<LogoutButtonProps>) {
  return (props: LogoutButtonProps) => (
    <LogoutButton {...defaultProps} {...props} />
  );
}

/**
 * Helper to check if user can logout all sessions
 */
export function canLogoutAllSessions(): boolean {
  const { isAdmin } = useRoles();
  return isAdmin;
}

export default LogoutButton;
