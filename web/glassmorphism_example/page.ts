import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Copy, Settings, TrendingUp, Calendar, Tag } from "lucide-react";
import { HyperparametersSection } from "@/components/HyperparametersSection";
import { MetricsSection } from "@/components/MetricsSection";

const Index = () => {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="min-h-screen bg-slate-900 relative">
      {/* Subtle background pattern for depth */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-purple-500/5 to-pink-500/5 opacity-50"></div>
      
      {/* Header */}
      <header className="relative border-b border-white/20 backdrop-blur-xl bg-black/10 shadow-2xl">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500/80 to-purple-600/80 rounded-lg flex items-center justify-center backdrop-blur-sm border border-white/20 shadow-lg">
                  <TrendingUp className="w-4 h-4 text-white" />
                </div>
                <h1 className="text-xl font-bold text-white">ML Experiments</h1>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-slate-300">
              <Calendar className="w-4 h-4" />
              <span>Jun 9, 2025 at 02:56 PM</span>
              <Badge variant="secondary" className="bg-green-500/20 text-green-300 border-green-400/30 backdrop-blur-sm">
                Private
              </Badge>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8 relative">
        {/* Project Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-3xl font-bold text-white mb-2">
                Digital Pattern Recognition
              </h2>
              <p className="text-slate-300 mb-4">
                Training my first CNN on MNIST digits to learn about convolutional layers and image classification
              </p>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="bg-blue-500/20 text-blue-300 border-blue-400/40 backdrop-blur-sm">
                  <Tag className="w-3 h-3 mr-1" />
                  junior_engineer
                </Badge>
                <Badge variant="outline" className="bg-purple-500/20 text-purple-300 border-purple-400/40 backdrop-blur-sm">
                  mnist
                </Badge>
                <Badge variant="outline" className="bg-green-500/20 text-green-300 border-green-400/40 backdrop-blur-sm">
                  cnn
                </Badge>
                <Badge variant="outline" className="bg-yellow-500/20 text-yellow-300 border-yellow-400/40 backdrop-blur-sm">
                  deep_learning
                </Badge>
                <Badge variant="outline" className="bg-orange-500/20 text-orange-300 border-orange-400/40 backdrop-blur-sm">
                  computer_vision
                </Badge>
                <Badge variant="outline" className="bg-red-500/20 text-red-300 border-red-400/40 backdrop-blur-sm">
                  beginner
                </Badge>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" className="bg-white/10 border-white/30 text-white hover:bg-white/20 backdrop-blur-md shadow-lg">
                <Copy className="w-4 h-4 mr-2" />
                Clone
              </Button>
              <Button variant="outline" size="sm" className="bg-white/10 border-white/30 text-white hover:bg-white/20 backdrop-blur-md shadow-lg">
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        </div>

        {/* Main Dashboard */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-black/20 border border-white/20 backdrop-blur-xl shadow-2xl">
            <TabsTrigger value="overview" className="data-[state=active]:bg-white/20 data-[state=active]:backdrop-blur-sm data-[state=active]:shadow-lg">
              Overview
            </TabsTrigger>
            <TabsTrigger value="logs" className="data-[state=active]:bg-white/20 data-[state=active]:backdrop-blur-sm data-[state=active]:shadow-lg">
              Logs
            </TabsTrigger>
            <TabsTrigger value="artifacts" className="data-[state=active]:bg-white/20 data-[state=active]:backdrop-blur-sm data-[state=active]:shadow-lg">
              Artifacts
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
              {/* Hyperparameters Section */}
              <HyperparametersSection />
              
              {/* Metrics Section */}
              <MetricsSection />
            </div>
          </TabsContent>

          <TabsContent value="logs">
            <Card className="bg-black/20 border-white/20 backdrop-blur-xl shadow-2xl">
              <CardHeader>
                <CardTitle className="text-white">Training Logs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-slate-300 font-mono text-sm space-y-2 bg-black/30 rounded-lg p-4 backdrop-blur-sm border border-white/10">
                  <div>Epoch 1/10 - loss: 0.2345 - accuracy: 0.9123 - val_loss: 0.1567 - val_accuracy: 0.9456</div>
                  <div>Epoch 2/10 - loss: 0.1892 - accuracy: 0.9234 - val_loss: 0.1234 - val_accuracy: 0.9567</div>
                  <div>Epoch 3/10 - loss: 0.1567 - accuracy: 0.9345 - val_loss: 0.1123 - val_accuracy: 0.9678</div>
                  <div className="text-green-400">Training completed successfully!</div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="artifacts">
            <Card className="bg-black/20 border-white/20 backdrop-blur-xl shadow-2xl">
              <CardHeader>
                <CardTitle className="text-white">Model Artifacts</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-slate-300">
                  <p>Model checkpoints, weights, and other artifacts will be displayed here.</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;
