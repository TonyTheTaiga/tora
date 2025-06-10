import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { BarChart3, TrendingUp, Database, Download } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts";

export const MetricsSection = () => {
  // Sample training data
  const trainingData = [
    { epoch: 1, loss: 0.2345, accuracy: 0.9123, val_loss: 0.1567, val_accuracy: 0.9456 },
    { epoch: 2, loss: 0.1892, accuracy: 0.9234, val_loss: 0.1234, val_accuracy: 0.9567 },
    { epoch: 3, loss: 0.1567, accuracy: 0.9345, val_loss: 0.1123, val_accuracy: 0.9678 },
    { epoch: 4, loss: 0.1345, accuracy: 0.9423, val_loss: 0.1089, val_accuracy: 0.9723 },
    { epoch: 5, loss: 0.1234, accuracy: 0.9489, val_loss: 0.1056, val_accuracy: 0.9756 },
    { epoch: 6, loss: 0.1156, accuracy: 0.9534, val_loss: 0.1023, val_accuracy: 0.9789 },
    { epoch: 7, loss: 0.1089, accuracy: 0.9567, val_loss: 0.0998, val_accuracy: 0.9812 },
    { epoch: 8, loss: 0.1023, accuracy: 0.9598, val_loss: 0.0976, val_accuracy: 0.9834 },
    { epoch: 9, loss: 0.0976, accuracy: 0.9623, val_loss: 0.0954, val_accuracy: 0.9856 },
    { epoch: 10, loss: 0.0934, accuracy: 0.9645, val_loss: 0.0934, val_accuracy: 0.9878 },
  ];

  const metrics = [
    { name: "Final Accuracy", value: "96.45%", change: "+2.3%", color: "text-green-400" },
    { name: "Final Loss", value: "0.0934", change: "-0.14", color: "text-green-400" },
    { name: "Val Accuracy", value: "98.78%", change: "+1.8%", color: "text-green-400" },
    { name: "Training Time", value: "12.4s", change: "-2.1s", color: "text-green-400" },
  ];

  return (
    <Card className="bg-black/20 border-white/20 backdrop-blur-xl shadow-2xl hover:bg-black/30 transition-all duration-300 hover:shadow-3xl">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-purple-400" />
            <span>Metrics</span>
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="bg-green-500/20 text-green-300 border-green-400/40 backdrop-blur-sm shadow-lg">
              Completed
            </Badge>
            <Button variant="ghost" size="sm" className="text-slate-300 hover:text-white hover:bg-white/20 backdrop-blur-sm">
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="chart" className="space-y-4">
          <TabsList className="bg-black/20 border border-white/20 w-full backdrop-blur-xl shadow-lg">
            <TabsTrigger value="chart" className="data-[state=active]:bg-white/20 data-[state=active]:backdrop-blur-sm data-[state=active]:shadow-lg flex-1">
              Chart
            </TabsTrigger>
            <TabsTrigger value="data" className="data-[state=active]:bg-white/20 data-[state=active]:backdrop-blur-sm data-[state=active]:shadow-lg flex-1">
              Data
            </TabsTrigger>
          </TabsList>

          <TabsContent value="chart" className="space-y-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-2 gap-3">
              {metrics.map((metric) => (
                <div key={metric.name} className="bg-white/5 rounded-lg p-3 hover:bg-white/10 transition-colors backdrop-blur-sm border border-white/10 shadow-lg">
                  <div className="text-xs text-slate-300 mb-1">{metric.name}</div>
                  <div className="flex items-center justify-between">
                    <span className="text-white font-semibold">{metric.value}</span>
                    <span className={`text-xs ${metric.color} flex items-center`}>
                      <TrendingUp className="w-3 h-3 mr-1" />
                      {metric.change}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Training Progress Chart */}
            <div className="space-y-4">
              <h4 className="text-white font-medium">Training Progress</h4>
              <div className="h-64 w-full bg-black/20 rounded-lg p-4 backdrop-blur-sm border border-white/10 shadow-inner">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trainingData}>
                    <defs>
                      <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                      </linearGradient>
                      <linearGradient id="valAccuracyGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                    <XAxis 
                      dataKey="epoch" 
                      stroke="#9ca3af"
                      fontSize={12}
                    />
                    <YAxis 
                      stroke="#9ca3af"
                      fontSize={12}
                      domain={[0.9, 1]}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(31, 41, 55, 0.8)', 
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                        color: '#f9fafb',
                        backdropFilter: 'blur(12px)',
                        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="accuracy"
                      stroke="#3b82f6"
                      fillOpacity={1}
                      fill="url(#accuracyGradient)"
                      strokeWidth={2}
                    />
                    <Area
                      type="monotone"
                      dataKey="val_accuracy"
                      stroke="#10b981"
                      fillOpacity={1}
                      fill="url(#valAccuracyGradient)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center space-x-6 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full shadow-lg"></div>
                  <span className="text-slate-300">Training Accuracy</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full shadow-lg"></div>
                  <span className="text-slate-300">Validation Accuracy</span>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="data">
            <div className="bg-white/5 rounded-lg p-4 backdrop-blur-sm border border-white/10 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <span className="text-white font-medium flex items-center space-x-2">
                  <Database className="w-4 h-4" />
                  <span>Training Data (0 of 42)</span>
                </span>
              </div>
              <div className="text-center py-12 text-slate-300">
                <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Select metrics from above to view and compare chart data</p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};
