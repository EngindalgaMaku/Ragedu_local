"use client";

import React, { useState, useEffect } from "react";
import { AdminUser } from "@/lib/admin-api";

interface UserModalProps {
  user: AdminUser | null;
  mode: "create" | "edit";
  onSave: (userData: any) => void;
  onClose: () => void;
}

const availableRoles = ["admin", "teacher", "student"];

export default function UserModal({
  user,
  mode,
  onSave,
  onClose,
}: UserModalProps) {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    first_name: "",
    last_name: "",
    password: "",
    confirmPassword: "",
    role_name: "student",
    is_active: true,
  });

  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (user && mode === "edit") {
      setFormData({
        username: user.username,
        email: user.email,
        first_name: user.first_name,
        last_name: user.last_name,
        password: "",
        confirmPassword: "",
        role_name: user.role_name,
        is_active: user.is_active,
      });
    } else {
      setFormData({
        username: "",
        email: "",
        first_name: "",
        last_name: "",
        password: "",
        confirmPassword: "",
        role_name: "student",
        is_active: true,
      });
    }
    setErrors({});
  }, [user, mode]);

  const validateForm = () => {
    const newErrors: Record<string, string> = {};

    // Username validation
    if (!formData.username.trim()) {
      newErrors.username = "Username is required";
    } else if (formData.username.length < 3) {
      newErrors.username = "Username must be at least 3 characters";
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!formData.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!emailRegex.test(formData.email)) {
      newErrors.email = "Please enter a valid email address";
    }

    // First name validation
    if (!formData.first_name.trim()) {
      newErrors.first_name = "First name is required";
    }

    // Last name validation
    if (!formData.last_name.trim()) {
      newErrors.last_name = "Last name is required";
    }

    // Password validation (only for create mode or if password is provided in edit mode)
    if (mode === "create" || formData.password) {
      if (!formData.password) {
        newErrors.password = "Password is required";
      } else if (formData.password.length < 6) {
        newErrors.password = "Password must be at least 6 characters";
      }

      if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = "Passwords do not match";
      }
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
      const userData = {
        ...(mode === "edit" && { id: user!.id }),
        ...(mode === "create" && { id: Date.now() }),
        username: formData.username,
        email: formData.email,
        first_name: formData.first_name,
        last_name: formData.last_name,
        role_name: formData.role_name,
        is_active: formData.is_active,
        // Always include password for create mode
        ...(mode === "create" && { password: formData.password }),
        // Include password only if it was provided in edit mode
        ...(mode === "edit" &&
          formData.password.trim() && { password: formData.password }),
      };

      onSave(userData);
    } catch (error) {
      console.error("Failed to save user:", error);
      setErrors({ submit: "Failed to save user. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    const checked =
      type === "checkbox" ? (e.target as HTMLInputElement).checked : undefined;

    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));

    // Clear error for this field when user starts typing
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: "" }));
    }
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
        <div className="inline-block align-bottom bg-white dark:bg-gray-800 rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
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
                {mode === "create" ? "Create New User" : "Edit User"}
              </h3>

              <div className="mt-6">
                <form onSubmit={handleSubmit} className="space-y-4">
                  {/* Username */}
                  <div>
                    <label
                      htmlFor="username"
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                    >
                      Username
                    </label>
                    <input
                      type="text"
                      id="username"
                      name="username"
                      value={formData.username}
                      onChange={handleInputChange}
                      className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white ${
                        errors.username
                          ? "border-red-300 dark:border-red-600"
                          : "border-gray-300 dark:border-gray-600"
                      }`}
                      placeholder="Enter username"
                    />
                    {errors.username && (
                      <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                        {errors.username}
                      </p>
                    )}
                  </div>

                  {/* Email */}
                  <div>
                    <label
                      htmlFor="email"
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                    >
                      Email
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white ${
                        errors.email
                          ? "border-red-300 dark:border-red-600"
                          : "border-gray-300 dark:border-gray-600"
                      }`}
                      placeholder="Enter email address"
                    />
                    {errors.email && (
                      <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                        {errors.email}
                      </p>
                    )}
                  </div>

                  {/* First Name & Last Name */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label
                        htmlFor="first_name"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                      >
                        First Name
                      </label>
                      <input
                        type="text"
                        id="first_name"
                        name="first_name"
                        value={formData.first_name}
                        onChange={handleInputChange}
                        className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white ${
                          errors.first_name
                            ? "border-red-300 dark:border-red-600"
                            : "border-gray-300 dark:border-gray-600"
                        }`}
                        placeholder="First name"
                      />
                      {errors.first_name && (
                        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                          {errors.first_name}
                        </p>
                      )}
                    </div>

                    <div>
                      <label
                        htmlFor="last_name"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                      >
                        Last Name
                      </label>
                      <input
                        type="text"
                        id="last_name"
                        name="last_name"
                        value={formData.last_name}
                        onChange={handleInputChange}
                        className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white ${
                          errors.last_name
                            ? "border-red-300 dark:border-red-600"
                            : "border-gray-300 dark:border-gray-600"
                        }`}
                        placeholder="Last name"
                      />
                      {errors.last_name && (
                        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                          {errors.last_name}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Password Fields */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label
                        htmlFor="password"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                      >
                        {mode === "create"
                          ? "Password"
                          : "New Password (optional)"}
                      </label>
                      <input
                        type="password"
                        id="password"
                        name="password"
                        value={formData.password}
                        onChange={handleInputChange}
                        className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white ${
                          errors.password
                            ? "border-red-300 dark:border-red-600"
                            : "border-gray-300 dark:border-gray-600"
                        }`}
                        placeholder={
                          mode === "create"
                            ? "Enter password"
                            : "Leave empty to keep current"
                        }
                      />
                      {errors.password && (
                        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                          {errors.password}
                        </p>
                      )}
                    </div>

                    <div>
                      <label
                        htmlFor="confirmPassword"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                      >
                        Confirm Password
                      </label>
                      <input
                        type="password"
                        id="confirmPassword"
                        name="confirmPassword"
                        value={formData.confirmPassword}
                        onChange={handleInputChange}
                        className={`mt-1 block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white ${
                          errors.confirmPassword
                            ? "border-red-300 dark:border-red-600"
                            : "border-gray-300 dark:border-gray-600"
                        }`}
                        placeholder="Confirm password"
                      />
                      {errors.confirmPassword && (
                        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                          {errors.confirmPassword}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Role */}
                  <div>
                    <label
                      htmlFor="role_name"
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                    >
                      Role
                    </label>
                    <select
                      id="role_name"
                      name="role_name"
                      value={formData.role_name}
                      onChange={handleInputChange}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                    >
                      {availableRoles.map((role) => (
                        <option key={role} value={role}>
                          {role.charAt(0).toUpperCase() + role.slice(1)}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Status */}
                  <div>
                    <div className="flex items-center">
                      <input
                        id="is_active"
                        name="is_active"
                        type="checkbox"
                        checked={formData.is_active}
                        onChange={handleInputChange}
                        className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                      />
                      <label
                        htmlFor="is_active"
                        className="ml-2 block text-sm text-gray-700 dark:text-gray-300"
                      >
                        Active User
                      </label>
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
                        "Create User"
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
