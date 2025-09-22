"use client";

import React from 'react';
import { Button } from "@/components/ui/button";
import { X, Maximize2, Minimize2 } from "lucide-react";

interface SidePanelOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
}

export function SidePanelOverlay({ isOpen, onClose, title = "Analysis Results", children }: SidePanelOverlayProps) {
  const [isMaximized, setIsMaximized] = React.useState(false);

  // Reset to half-screen when opened
  React.useEffect(() => {
    if (isOpen) {
      setIsMaximized(false);
    }
  }, [isOpen]);

  console.log('SidePanelOverlay: isOpen =', isOpen, ', isMaximized =', isMaximized);

  if (!isOpen) return null;

  return (
    <>
      {/* Debug overlay - always visible when isOpen */}
      {isOpen && (
        <div className="fixed top-4 left-4 bg-red-500 text-white p-2 z-[9999] text-xs">
          Overlay Active (isMaximized: {isMaximized.toString()})
        </div>
      )}
      
      {/* Backdrop - only covers the non-panel area */}
      <div 
        className={`
          fixed top-0 left-0 h-full bg-black/40 z-40 transition-all duration-300 ease-in-out
          ${isMaximized ? 'w-0' : 'w-[50vw]'}
          ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        `}
        onClick={onClose}
      />
      
      {/* Backdrop for maximized state */}
      {isMaximized && isOpen && (
        <div 
          className="fixed inset-0 bg-black/20 z-40"
          onClick={onClose}
        />
      )}
      
      {/* Side Panel */}
      <div 
        className={`
          fixed top-0 right-0 h-full bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 z-50 shadow-2xl
          transition-all duration-300 ease-in-out
          ${isMaximized ? 'w-full' : 'w-[50vw]'}
          ${isOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border bg-card/50">
          <h2 className="text-lg font-semibold text-foreground">{title}</h2>
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
      </div>
    </>
  );
}