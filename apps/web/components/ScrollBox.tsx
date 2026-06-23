"use client";

export function ScrollBox({
  children,
  maxHeight = 420,
  className = "",
}: {
  children: React.ReactNode;
  maxHeight?: number | string;
  className?: string;
}) {
  return (
    <div
      className={`scroll-box ${className}`.trim()}
      style={{ maxHeight: typeof maxHeight === "number" ? `${maxHeight}px` : maxHeight }}
    >
      {children}
    </div>
  );
}
