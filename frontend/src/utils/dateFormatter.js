/**
 * Formats a date string to a human-readable format
 * @param {string} dateString - The date string to format
 * @returns {string} Formatted date string or 'N/A' if no date provided
 */
export const formatDate = (dateString) => {
  if (!dateString) return 'N/A';
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

/**
 * Formats a date string to a short format (date only)
 * @param {string} dateString - The date string to format
 * @returns {string} Formatted date string or 'N/A' if no date provided
 */
export const formatDateShort = (dateString) => {
  if (!dateString) return 'N/A';
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

/**
 * Formats a date string to show relative time (e.g., "2 hours ago")
 * @param {string} dateString - The date string to format
 * @returns {string} Relative time string or 'N/A' if no date provided
 */
export const formatRelativeTime = (dateString) => {
  if (!dateString) return 'N/A';

  const now = new Date();
  const date = new Date(dateString);
  const diffInMs = now - date;
  const diffInMinutes = Math.floor(diffInMs / (1000 * 60));
  const diffInHours = Math.floor(diffInMinutes / 60);
  const diffInDays = Math.floor(diffInHours / 24);

  if (diffInMinutes < 1) return 'Just now';
  if (diffInMinutes < 60) return `${diffInMinutes} minute${diffInMinutes > 1 ? 's' : ''} ago`;
  if (diffInHours < 24) return `${diffInHours} hour${diffInHours > 1 ? 's' : ''} ago`;
  if (diffInDays < 7) return `${diffInDays} day${diffInDays > 1 ? 's' : ''} ago`;

  return formatDateShort(dateString);
};
