import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Copy, ChevronDown, Settings2 } from "lucide-react";
import { useState } from "react";

export const HyperparametersSection = () => {
  const [showAll, setShowAll] = useState(false);

  const hyperparameters = [
    { key: "dataset", value: "MNIST", type: "string" },
    { key: "n_samples_requested", value: "10000", type: "number" },
    { key: "model_complexity", value: "simple", type: "string" },
    { key: "optimizer", value: "Adam", type: "string" },
    { key: "learning_rate", value: "0.001", type: "number" },
    { key: "batch_size", value: "64", type: "number" },
    { key: "epochs", value: "10", type: "number" },
    { key: "dropout_rate", value: "0.2", type: "number" },
    { key: "conv_layers", value: "3", type: "number" },
    { key: "filters", value: "[32, 64, 128]", type: "array" },
    { key: "kernel_size", value: "3", type: "number" },
    { key: "activation", value: "relu", type: "string" },
  ];

  const visibleParams = showAll ? hyperparameters : hyperparameters.slice(0, 7);

  const copyToClipboard = (value: string) => {
    navigator.clipboard.writeText(value);
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "string": return "bg-blue-500/20 text-blue-300 border-blue-400/40 backdrop-blur-sm";
      case "number": return "bg-green-500/20 text-green-300 border-green-400/40 backdrop-blur-sm";
      case "array": return "bg-purple-500/20 text-purple-300 border-purple-400/40 backdrop-blur-sm";
      default: return "bg-slate-500/20 text-slate-300 border-slate-400/40 backdrop-blur-sm";
    }
  };

  return (
    <Card className="bg-black/20 border-white/20 backdrop-blur-xl shadow-2xl hover:bg-black/30 transition-all duration-300 hover:shadow-3xl">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center space-x-2">
            <Settings2 className="w-5 h-5 text-blue-400" />
            <span>Hyperparameters</span>
          </CardTitle>
          <Badge variant="outline" className="bg-blue-500/20 text-blue-300 border-blue-400/40 backdrop-blur-sm shadow-lg">
            {hyperparameters.length} params
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {visibleParams.map((param) => (
          <div
            key={param.key}
            className="flex items-center justify-between p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-all duration-200 group backdrop-blur-sm border border-white/10 shadow-lg hover:shadow-xl"
          >
            <div className="flex items-center space-x-3 flex-1">
              <Badge variant="outline" className={`text-xs ${getTypeColor(param.type)} shadow-sm`}>
                {param.type}
              </Badge>
              <span className="text-slate-200 font-medium">{param.key}</span>
            </div>
            <div className="flex items-center space-x-2">
              <code className="text-white font-mono bg-black/40 px-2 py-1 rounded text-sm backdrop-blur-sm border border-white/20 shadow-inner">
                {param.value}
              </code>
              <Button
                variant="ghost"
                size="sm"
                className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8 p-0 hover:bg-white/20 backdrop-blur-sm"
                onClick={() => copyToClipboard(param.value)}
              >
                <Copy className="w-3 h-3 text-slate-300" />
              </Button>
            </div>
          </div>
        ))}
        
        {hyperparameters.length > 7 && (
          <Button
            variant="ghost"
            onClick={() => setShowAll(!showAll)}
            className="w-full text-slate-300 hover:text-white hover:bg-white/10 mt-4 backdrop-blur-sm border border-white/10 shadow-lg"
          >
            <ChevronDown className={`w-4 h-4 mr-2 transition-transform ${showAll ? 'rotate-180' : ''}`} />
            {showAll ? 'Show less' : `Show ${hyperparameters.length - 7} more hyperparameters`}
          </Button>
        )}
      </CardContent>
    </Card>
  );
};

