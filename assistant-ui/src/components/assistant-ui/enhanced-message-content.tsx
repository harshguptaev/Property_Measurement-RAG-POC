"use client";

import React from 'react';
import { RoofPitchResultButton } from "@/components/RoofPitchResultButton";

interface MessageContentProps {
  text: string;
  hasRoofPitchData?: boolean;
  roofPitchData?: any[];
}

export function EnhancedMessageContent({ text, hasRoofPitchData, roofPitchData }: MessageContentProps) {
  // Check if this is a roof pitch response that should show the result button
  const shouldShowResultButton = hasRoofPitchData && roofPitchData && roofPitchData.length > 0;

  return (
    <div className="space-y-4">
      {/* Render the markdown text */}
      <div 
        className="prose prose-sm max-w-none"
        dangerouslySetInnerHTML={{ 
          __html: text
            .replace(/\n/g, '<br/>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/# (.*)/g, '<h1 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">$1</h1>')
            .replace(/## (.*)/g, '<h2 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.75rem;">$2</h2>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        }} 
      />
      
      {/* Show result button if roof pitch data is available */}
      {shouldShowResultButton && (
        <RoofPitchResultButton 
          roofPitchData={roofPitchData}
          propertiesCount={roofPitchData.length}
        />
      )}
    </div>
  );
}