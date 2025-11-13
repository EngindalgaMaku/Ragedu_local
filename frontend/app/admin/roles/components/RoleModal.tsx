"use client";

import React, { useState, useEffect } from "react";

interface Role {
  id: number;
  name: string;
  description: string;
  permissions: Record<string, string[]>;
  user_count?: number;
  created_at: string;
  updated_at: string;
}

interface RoleModalProps {
  role: Role | null;
  mode: "create" | "edit";
  onSave: (role: Role) => void;
  onClose: () => void;
}

// Available resources and actions
const AVAILABLE_RESOURCES = {
  users: ["create", "read", "update", "delete"],
  roles: ["create", "read", "update", "delete"],
  sessions: ["create", "read", "update", "delete"],
  documents: ["create", "read", "update", "delete"],
  students: ["read", "update"],
  system: ["admin", "configure", "monitor"],
};

export default function RoleModal({
  role,
  mode,
  onSave,
  onClose,
}: RoleModalProps) {
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    permissions: {} as Record<string, string[]>,
  });

  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (role && mode === "edit") {
      setFormData({
        name: role.name,
        description: role.description,
        permissions: { ...role.permissions },
      });
    } else {
      setFormData({
        name: "",
        description: "",
        permissions: {},
      });
    }
    setErrors({});
  }, [role, mode]);

  const validateForm = () => {
    const newErrors: Record<string, string> = {};

    // Name validation
    if (!formData.name.trim()) {
      newErrors.name = "Role name is required";
    } else if (formData.name.length < 2) {
      newErrors.name = "Role name must be at least 2 characters";
    } else if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(formData.name)) {
      newErrors.name =
        "Role name can only contain letters, numbers, and underscores, and must start with a letter";
    }

    // Description validation
    if (!formData.description.trim()) {
      newErrors.description = "Description is required";
    } else if (formData.description.length < 10) {
      newErrors.description = "Description must be at least 10 characters";
    }

    // Permissions validation
    const hasPermissions = Object.values(formData.permissions).some(
      (actions) => actions.length > 0
    );
    if (!hasPermissions) {
      newErrors.permissions = "At least one permission must be selected";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      const roleData: Role = {
        id: role?.id || Date.now(),
        name: formData.name.toLowerCase(),
        description: formData.description,
        permissions: formData.permissions,
        created_at: role?.created_at || new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));

      onSave(roleData);
    } catch (error) {
      console.error("Failed to save role:", error);
      setErrors({ submit: "Failed to save role. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));

    // Clear error for this field when user starts typing
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: "" }));
    }
  };

  const handlePermissionToggle = (resource: string, action: string) => {
    setFormData((prev) => {
      const currentActions = prev.permissions[resource] || [];
      const updatedActions = currentActions.includes(action)
        ? currentActions.filter((a) => a !== action)
        : [...currentActions, action];

      const updatedPermissions = { ...prev.permissions };
      if (updatedActions.length === 0) {
        delete updatedPermissions[resource];
      } else {
        updatedPermissions[resource] = updatedActions;
      }

      return {
        ...prev,
        permissions: updatedPermissions,
      };
    });

    // Clear permissions error when user makes selection
    if (errors.permissions) {
      setErrors((prev) => ({ ...prev, permissions: "" }));
    }
  };

  const handleResourceToggle = (resource: string) => {
    const allActions =
      AVAILABLE_RESOURCES[resource as keyof typeof AVAILABLE_RESOURCES];
    const currentActions = formData.permissions[resource] || [];
    const hasAllActions = allActions.every((action) =>
      currentActions.includes(action)
    );

    if (hasAllActions) {
      // Remove all actions for this resource
      setFormData((prev) => {
        const updatedPermissions = { ...prev.permissions };
        delete updatedPermissions[resource];
        return { ...prev, permissions: updatedPermissions };
      });
    } else {
      // Add all actions for this resource
      setFormData((prev) => ({
        ...prev,
        permissions: {
          ...prev.permissions,
          [resource]: allActions,
        },
      }));
    }

    if (errors.permissions) {
      setErrors((prev) => ({ ...prev, permissions: "" }));
    }
  };

  const isResourceFullySelected = (resource: string) => {
    const allActions =
      AVAILABLE_RESOURCES[resource as keyof typeof AVAILABLE_RESOURCES];
    const currentActions = formData.permissions[resource] || [];
    return allActions.every((action) => currentActions.includes(action));
  };

  const isResourcePartiallySelected = (resource: string) => {
    const currentActions = formData.permissions[resource] || [];
    return currentActions.length > 0 && !isResourceFullySelected(resource);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        {/* Background overlay */}
        <div
          className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"
          onClick={onClose}
        ></div>

        {/* Modal */}
        <div className="inline-block align-bottom bg-white dark:bg-gray-800 rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-2xl sm:w-full sm:p-6">
          <div className="absolute top-0 right-0 pt-4 pr-4">
            <button
              type="button"
              className="bg-white dark:bg-gray-800 rounded-md text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              onClick={onClose}
            >
              <span className="sr-only">Close</span>
              <svg
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <div className="sm:flex sm:items-start">
            <div className="mt-3 text-center sm:mt-0 sm:text-left w-full">
              <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">
                {mode === "create" ? "Create New Role" : "Edit Role"}
              </h3>

              <div className="mt-6">
                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Name */}
                  <div>
                    <label
                      htmlFor="name"
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                    >
                      Role Name
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      disabled={mode === "edit"} // Don't allow editing role names
                      className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white ${
                        errors.name
                          ? "border-red-300 dark:border-red-600"
                          : "border-gray-300 dark:border-gray-600"
                      } ${
                        mode === "edit"
                          ? "bg-gray-100 dark:bg-gray-600 cursor-not-allowed"
                          : ""
                      }`}
                      placeholder="Enter role name (e.g., teacher, student)"
                    />
                    {errors.name && (
                      <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                        {errors.name}
                      </p>
                    )}
                    {mode === "edit" && (
                      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                        Role name cannot be changed after creation
                      </p>
                    )}
                  </div>

                  {/* Description */}
                  <div>
                    <label
                      htmlFor="description"
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                    >
                      Description
                    </label>
                    <textarea
                      id="description"
                      name="description"
                      rows={3}
                      value={formData.description}
                      onChange={handleInputChange}
                      className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white ${
                        errors.description
                          ? "border-red-300 dark:border-red-600"
                          : "border-gray-300 dark:border-gray-600"
                      }`}
                      placeholder="Describe what this role can do..."
                    />
                    {errors.description && (
                      <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                        {errors.description}
                      </p>
                    )}
                  </div>

                  {/* Permissions */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                      Permissions
                    </label>
                    {errors.permissions && (
                      <p className="mb-3 text-sm text-red-600 dark:text-red-400">
                        {errors.permissions}
                      </p>
                    )}

                    <div className="space-y-4">
                      {Object.entries(AVAILABLE_RESOURCES).map(
                        ([resource, actions]) => {
                          const isFullySelected =
                            isResourceFullySelected(resource);
                          const isPartiallySelected =
                            isResourcePartiallySelected(resource);

                          return (
                            <div
                              key={resource}
                              className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                            >
                              <div className="flex items-center justify-between mb-3">
                                <label className="flex items-center cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={isFullySelected}
                                    ref={(el) => {
                                      if (el)
                                        el.indeterminate = isPartiallySelected;
                                    }}
                                    onChange={() =>
                                      handleResourceToggle(resource)
                                    }
                                    className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                                  />
                                  <span className="ml-2 text-sm font-medium text-gray-900 dark:text-white capitalize">
                                    {resource}
                                  </span>
                                </label>
                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                  {formData.permissions[resource]?.length || 0}{" "}
                                  / {actions.length} selected
                                </span>
                              </div>

                              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                                {actions.map((action) => (
                                  <label
                                    key={action}
                                    className="flex items-center cursor-pointer"
                                  >
                                    <input
                                      type="checkbox"
                                      checked={
                                        formData.permissions[
                                          resource
                                        ]?.includes(action) || false
                                      }
                                      onChange={() =>
                                        handlePermissionToggle(resource, action)
                                      }
                                      className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                                    />
                                    <span className="ml-2 text-sm text-gray-600 dark:text-gray-400 capitalize">
                                      {action}
                                    </span>
                                  </label>
                                ))}
                              </div>
                            </div>
                          );
                        }
                      )}
                    </div>
                  </div>

                  {/* Submit Error */}
                  {errors.submit && (
                    <div className="text-sm text-red-600 dark:text-red-400 text-center">
                      {errors.submit}
                    </div>
                  )}

                  {/* Actions */}
                  <div className="mt-6 flex justify-end space-x-3">
                    <button
                      type="button"
                      onClick={onClose}
                      disabled={loading}
                      className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={loading}
                      className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? (
                        <div className="flex items-center">
                          <svg
                            className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                          >
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                            ></circle>
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            ></path>
                          </svg>
                          Saving...
                        </div>
                      ) : mode === "create" ? (
                        "Create Role"
                      ) : (
                        "Save Changes"
                      )}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
