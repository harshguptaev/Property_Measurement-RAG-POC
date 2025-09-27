"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface PropertyImage {
  type: string;
  title: string;
  description: string;
  image_path: string;
}

interface PropertyImageData {
  property_id: string;
  property_address: string;
  images: {
    measurement_images: PropertyImage[];
    property_views: PropertyImage[];
    roof_analysis: PropertyImage[];
  };
}

interface ImageGalleryData {
  status: string;
  properties_found: number;
  image_data: PropertyImageData[];
}

interface ImageGalleryViewerProps {
  onClose?: () => void;
}

export function ImageGalleryViewer({ onClose }: ImageGalleryViewerProps) {
  const [imageData, setImageData] = useState<ImageGalleryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedProperty, setSelectedProperty] = useState<string | null>(null);

  useEffect(() => {
    // First try to load data from sessionStorage
    const sessionData = sessionStorage.getItem('allImagesData');
    if (sessionData) {
      try {
        const parsedData = JSON.parse(sessionData);
        setImageData({
          status: 'success',
          properties_found: parsedData.length,
          image_data: parsedData
        });
        setLoading(false);
        if (parsedData.length > 0) {
          setSelectedProperty(parsedData[0].property_id);
        }
        return;
      } catch (err) {
        console.error('Error parsing session data:', err);
      }
    }
    
    // Fallback to API
    fetchImageData();
  }, []);

  const fetchImageData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const backendUrl =
        process.env.NEXT_PUBLIC_RAG_BACKEND_URL ||
        process.env.RAG_BACKEND_URL ||
        "http://localhost:8001";
      
      const response = await fetch(`${backendUrl}/all-images-data`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setImageData(data);
      if (data.image_data.length > 0) {
        setSelectedProperty(data.image_data[0].property_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error fetching image data:', err);
    } finally {
      setLoading(false);
    }
  };

  const getImageUrl = (imagePath: string) => {
    const backendUrl =
      process.env.NEXT_PUBLIC_RAG_BACKEND_URL ||
      process.env.RAG_BACKEND_URL ||
      "http://localhost:8001";
    
    const cleanPath = imagePath.replace(/^extracted_images\//, '');
    return `${backendUrl}/images/${cleanPath}`;
  };

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center min-h-[400px]">
        <div className="animate-pulse text-center">
          <div className="h-6 bg-gray-200 rounded w-48 mb-4 mx-auto"></div>
          <div className="space-y-2">
            <div className="h-24 bg-gray-200 rounded"></div>
            <div className="h-24 bg-gray-200 rounded"></div>
            <div className="h-24 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center">
        <div className="border border-red-200 bg-red-50 p-4 rounded-md">
          <p className="text-red-700">Error loading image data: {error}</p>
          <Button onClick={fetchImageData} className="mt-2" variant="outline" size="sm">
            Retry
          </Button>
        </div>
      </div>
    );
  }

  if (!imageData || imageData.properties_found === 0) {
    return (
      <div className="p-6 text-center">
        <p className="text-muted-foreground">No image data found.</p>
      </div>
    );
  }

  const selectedPropertyData = imageData.image_data.find(p => p.property_id === selectedProperty);

  return (
    <div className="flex flex-col h-full">
      {/* Property Selector */}
      <div className="p-4 border-b">
        <div className="flex flex-wrap gap-2">
          {imageData.image_data.map((property) => (
            <Button
              key={property.property_id}
              variant={selectedProperty === property.property_id ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedProperty(property.property_id)}
              className="text-xs"
            >
              {property.property_id}
            </Button>
          ))}
        </div>
        {selectedPropertyData && (
          <p className="text-sm text-muted-foreground mt-2">
            {selectedPropertyData.property_address}
          </p>
        )}
      </div>

      {/* Image Content */}
      {selectedPropertyData && (
        <div className="flex-1 overflow-auto p-4">
          <Tabs defaultValue="measurement" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="measurement">Measurements</TabsTrigger>
              <TabsTrigger value="views">Property Views</TabsTrigger>
              <TabsTrigger value="analysis">Roof Analysis</TabsTrigger>
            </TabsList>
            
            <TabsContent value="measurement" className="space-y-4">
              <div className="grid gap-4">
                {selectedPropertyData.images.measurement_images.map((image, index) => (
                  <Card key={index}>
                    <CardHeader>
                      <CardTitle className="text-sm">{image.title}</CardTitle>
                      <p className="text-xs text-muted-foreground">{image.description}</p>
                    </CardHeader>
                    <CardContent>
                      <img
                        src={getImageUrl(image.image_path)}
                        alt={image.title}
                        className="w-full h-auto rounded-md border"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='200'%3E%3Crect width='100%25' height='100%25' fill='%23f3f4f6'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' font-family='sans-serif' font-size='14' fill='%236b7280'%3EImage not available%3C/text%3E%3C/svg%3E";
                        }}
                      />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
            
            <TabsContent value="views" className="space-y-4">
              <div className="grid gap-4">
                {selectedPropertyData.images.property_views.map((image, index) => (
                  <Card key={index}>
                    <CardHeader>
                      <CardTitle className="text-sm">{image.title}</CardTitle>
                      <p className="text-xs text-muted-foreground">{image.description}</p>
                    </CardHeader>
                    <CardContent>
                      <img
                        src={getImageUrl(image.image_path)}
                        alt={image.title}
                        className="w-full h-auto rounded-md border"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='200'%3E%3Crect width='100%25' height='100%25' fill='%23f3f4f6'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' font-family='sans-serif' font-size='14' fill='%236b7280'%3EImage not available%3C/text%3E%3C/svg%3E";
                        }}
                      />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
            
            <TabsContent value="analysis" className="space-y-4">
              <div className="grid gap-4">
                {selectedPropertyData.images.roof_analysis.map((image, index) => (
                  <Card key={index}>
                    <CardHeader>
                      <CardTitle className="text-sm">{image.title}</CardTitle>
                      <p className="text-xs text-muted-foreground">{image.description}</p>
                    </CardHeader>
                    <CardContent>
                      <img
                        src={getImageUrl(image.image_path)}
                        alt={image.title}
                        className="w-full h-auto rounded-md border"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='200'%3E%3Crect width='100%25' height='100%25' fill='%23f3f4f6'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' font-family='sans-serif' font-size='14' fill='%236b7280'%3EImage not available%3C/text%3E%3C/svg%3E";
                        }}
                      />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      )}
    </div>
  );
}