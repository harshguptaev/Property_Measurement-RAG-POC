"use client";

import React from 'react';
import { Button } from "@/components/ui/button";
import { X, Maximize2, Minimize2 } from "lucide-react";

interface SidePanelOverlayProps {
  readonly isOpen: boolean;
  readonly onClose: () => void;
  readonly title?: string;
  readonly children: React.ReactNode;
}

export function SidePanelOverlay({ isOpen, onClose, title = "Analysis Results", children }: SidePanelOverlayProps) {
  const [isMaximized, setIsMaximized] = React.useState(false);

  // Reset to half-screen when opened
  React.useEffect(() => {
    if (isOpen) {
      setIsMaximized(false);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <button 
        className={`
          fixed inset-0 bg-black/40 z-40 transition-all duration-300 ease-in-out border-none cursor-pointer
          ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        `}
        onClick={onClose}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            onClose();
          }
        }}
        aria-label="Close overlay"
      />
      
      {/* Side Panel */}
      <dialog 
        className={`
          fixed top-0 right-0 h-full bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 z-50 shadow-2xl
          transition-all duration-300 ease-in-out max-w-none max-h-none m-0 p-0
          ${isMaximized ? 'w-full' : 'w-[50vw]'}
          ${isOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
        open={isOpen}
        aria-labelledby="panel-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border bg-card/50">
          <h2 id="panel-title" className="text-lg font-semibold text-foreground">{title}</h2>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMaximized(!isMaximized)}
              className="h-8 w-8 p-0"
            >
              {isMaximized ? (
                <Minimize2 className="h-4 w-4" />
              ) : (
                <Maximize2 className="h-4 w-4" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Content */}
        <div className="h-[calc(100vh-60px)] overflow-auto">
          {children}
        </div>
      </dialog>
    </>
  );
}