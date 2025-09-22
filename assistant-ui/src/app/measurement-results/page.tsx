"use client";

import React from 'react';
import { MeasurementViewer } from '@/components/MeasurementViewer';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import { useRouter, useSearchParams } from 'next/navigation';

export default function MeasurementResultsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const measurementType = searchParams.get('type') || 'lengths';

  const measurementTitles = {
    'lengths': 'Length Measurements Analysis',
    'rafters': 'Rafter Analysis Results', 
    'area': 'Roof Area Analysis',
    'azimuth': 'Roof Orientation Analysis'
  };

  const measurementDescriptions = {
    'lengths': 'Detailed ridge, hip, valley, and eave measurements',
    'rafters': 'Comprehensive rafter length calculations and analysis',
    'area': 'Complete roof area measurements and breakdowns',
    'azimuth': 'Roof facet orientations and azimuth data'
  };

  const title = measurementTitles[measurementType as keyof typeof measurementTitles] || 'Measurement Analysis';
  const description = measurementDescriptions[measurementType as keyof typeof measurementDescriptions] || 'Property measurement data';

  return (
    <div className="h-screen overflow-hidden flex flex-col">
      {/* Header */}
      <div className="bg-background border-b border-border p-4 flex items-center gap-4 flex-shrink-0">
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={() => router.back()}
          className="flex items-center gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Chat
        </Button>
        <div>
          <h1 className="text-lg font-semibold">{title}</h1>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      </div>

      {/* Main Content - Non-scrollable container */}
      <div className="flex-1 overflow-hidden">
        <MeasurementViewer measurementType={measurementType} />
      </div>
    </div>
  );
}