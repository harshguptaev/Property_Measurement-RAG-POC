'use client';

import React, { useState, useEffect } from 'react';
import { SidePanelOverlay } from './SidePanelOverlay';
import { ImageGalleryViewer } from './ImageGalleryViewer';

export function ImageGalleryManager() {
  const [isOpen, setIsOpen] = useState(false);
  const [imageData, setImageData] = useState(null);

  useEffect(() => {
    // Listen for image gallery events
    const handleShowImageGallery = (event: CustomEvent) => {
      console.log('ImageGalleryManager: Received showImageGallery event', event.detail);
      setImageData(event.detail);
      setIsOpen(true);
    };

    window.addEventListener('showImageGallery', handleShowImageGallery as EventListener);

    return () => {
      window.removeEventListener('showImageGallery', handleShowImageGallery as EventListener);
    };
  }, []);

  const handleClose = () => {
    console.log('ImageGalleryManager: Closing overlay');
    setIsOpen(false);
    setImageData(null);
  };

  console.log('ImageGalleryManager: isOpen =', isOpen, ', imageData =', !!imageData);

  return (
    <SidePanelOverlay isOpen={isOpen} onClose={handleClose} title="Property Image Gallery">
      <ImageGalleryViewer onClose={handleClose} />
    </SidePanelOverlay>
  );
}

export default ImageGalleryManager;