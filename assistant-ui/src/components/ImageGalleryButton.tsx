'use client';

import React from 'react';
import { Button } from './ui/button';
import { Images, Camera, Eye } from 'lucide-react';

interface ImageGalleryButtonProps {
  imageData?: any;
  className?: string;
}

export const ImageGalleryButton: React.FC<ImageGalleryButtonProps> = ({ 
  imageData, 
  className = "" 
}) => {
  const handleClick = () => {
    console.log('ImageGalleryButton: Button clicked, imageData =', imageData);
    if (imageData) {
      // Store image data in sessionStorage for the gallery
      sessionStorage.setItem('imageGalleryData', JSON.stringify(imageData));
      
      // Emit custom event to show image gallery overlay
      const event = new CustomEvent('showImageGallery', { detail: imageData });
      console.log('ImageGalleryButton: Dispatching showImageGallery event', event);
      window.dispatchEvent(event);
    } else {
      console.log('ImageGalleryButton: No imageData available');
    }
  };

  const getButtonContent = () => {
    if (imageData && Array.isArray(imageData)) {
      const totalImages = imageData.reduce((sum, property) => {
        return sum + 
          (property.measurement_images?.length || 0) + 
          (property.property_views?.length || 0) + 
          (property.roof_analysis?.length || 0);
      }, 0);

      return (
        <>
          <Images className="w-4 h-4" />
          View Image Gallery ({totalImages} images)
        </>
      );
    }

    return (
      <>
        <Camera className="w-4 h-4" />
        View Property Images
      </>
    );
  };

  return (
    <Button
      onClick={handleClick}
      variant="outline"
      size="sm"
      className={`flex items-center gap-2 ${className}`}
    >
      {getButtonContent()}
    </Button>
  );
};

export default ImageGalleryButton;