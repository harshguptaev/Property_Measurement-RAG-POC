'use client';

import React, { useState, useEffect } from 'react';
import { SidePanelOverlay } from './SidePanelOverlay';
import { MeasurementViewer } from './MeasurementViewer';

interface MeasurementOverlayData {
  measurementType: string;
  measurementData: any[];
  title: string;
  description: string;
}

export function MeasurementOverlayManager() {
  const [isOpen, setIsOpen] = useState(false);
  const [overlayData, setOverlayData] = useState<MeasurementOverlayData | null>(null);

  useEffect(() => {
    const handleShowMeasurementOverlay = (event: CustomEvent) => {
      console.log('MeasurementOverlayManager: Received showMeasurementOverlay event', event.detail);
      setOverlayData(event.detail);
      setIsOpen(true);
    };

    window.addEventListener('showMeasurementOverlay', handleShowMeasurementOverlay as EventListener);

    return () => {
      window.removeEventListener('showMeasurementOverlay', handleShowMeasurementOverlay as EventListener);
    };
  }, []);

  const handleClose = () => {
    console.log('MeasurementOverlayManager: Closing overlay');
    setIsOpen(false);
    setOverlayData(null);
  };

  console.log('MeasurementOverlayManager: isOpen =', isOpen, ', overlayData =', !!overlayData);

  return (
    <SidePanelOverlay 
      isOpen={isOpen} 
      onClose={handleClose} 
      title={overlayData?.title || 'Measurement Analysis'}
    >
      <div className="h-full">
        {overlayData && (
          <>
            <div className="p-4 border-b border-border">
              <p className="text-sm text-muted-foreground">{overlayData.description}</p>
            </div>
            <div className="flex-1 overflow-hidden">
              <MeasurementViewer measurementType={overlayData.measurementType} />
            </div>
          </>
        )}
      </div>
    </SidePanelOverlay>
  );
}

export default MeasurementOverlayManager;
