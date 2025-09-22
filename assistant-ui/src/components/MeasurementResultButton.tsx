"use client";

import React from 'react';
import { Button } from "@/components/ui/button";
import { BarChart3, ExternalLink, Ruler, Building, Calculator, Compass } from "lucide-react";

interface MeasurementResultButtonProps {
  readonly measurementData: any[];
  readonly measurementType: string;
  readonly propertiesCount: number;
}

export function MeasurementResultButton({ measurementData, measurementType, propertiesCount }: MeasurementResultButtonProps) {

  const measurementConfig = {
    'roof_pitch': {
      icon: BarChart3,
      title: 'Roof Pitch Analysis Ready',
      description: 'detailed pitch data and diagrams',
      overlayTitle: 'Roof Pitch Analysis',
      overlayDescription: 'Detailed pitch data and diagrams',
      storageKey: 'roofPitchData',
      route: '/roof-pitch-results'
    },
    'lengths': {
      icon: Ruler,
      title: 'Length Measurements Ready',
      description: 'ridge, hip, valley, and eave measurements',
      overlayTitle: 'Length Measurements Analysis',
      overlayDescription: 'Detailed ridge, hip, valley, and eave measurements',
      storageKey: 'lengthsData',
      route: '/measurement-results?type=lengths'
    },
    'rafters': {
      icon: Building,
      title: 'Rafter Analysis Ready',
      description: 'detailed rafter length measurements',
      overlayTitle: 'Rafter Analysis Results',
      overlayDescription: 'Comprehensive rafter length calculations and analysis',
      storageKey: 'raftersData',
      route: '/measurement-results?type=rafters'
    },
    'area': {
      icon: Calculator,
      title: 'Area Analysis Ready',
      description: 'roof area measurements and breakdowns',
      overlayTitle: 'Roof Area Analysis',
      overlayDescription: 'Complete roof area measurements and breakdowns',
      storageKey: 'areaData',
      route: '/measurement-results?type=area'
    },
    'azimuth': {
      icon: Compass,
      title: 'Orientation Analysis Ready',
      description: 'roof facet orientations and azimuth data',
      overlayTitle: 'Roof Orientation Analysis',
      overlayDescription: 'Roof facet orientations and azimuth data',
      storageKey: 'azimuthData',
      route: '/measurement-results?type=azimuth'
    }
  };

  const config = measurementConfig[measurementType as keyof typeof measurementConfig] || measurementConfig['roof_pitch'];
  const IconComponent = config.icon;

  const handleViewResults = () => {
    // Store the measurement data in sessionStorage for fallback
    sessionStorage.setItem(config.storageKey, JSON.stringify(measurementData));
    sessionStorage.setItem('measurementType', measurementType);
    
    // Dispatch event to show measurement overlay
    const event = new CustomEvent('showMeasurementOverlay', { 
      detail: {
        measurementType,
        measurementData,
        title: config.overlayTitle,
        description: config.overlayDescription
      }
    });
    console.log('MeasurementResultButton: Dispatching showMeasurementOverlay event', event);
    window.dispatchEvent(event);
  };

  return (
    <div className="mt-4 p-4 border border-border rounded-lg bg-card/50">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <IconComponent className="h-5 w-5 text-primary" />
          <div>
            <h4 className="font-medium text-sm">{config.title}</h4>
            <p className="text-xs text-muted-foreground">
              Found {propertiesCount} properties with {config.description}
            </p>
          </div>
        </div>
        <Button 
          onClick={handleViewResults}
          variant="default" 
          size="sm"
          className="flex items-center gap-2"
        >
          <ExternalLink className="h-4 w-4" />
          View Results
        </Button>
      </div>
    </div>
  );
}