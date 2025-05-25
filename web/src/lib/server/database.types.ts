export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export type Database = {
  public: {
    Tables: {
      api_keys: {
        Row: {
          created_at: string;
          id: string;
          key_hash: string;
          last_used: string;
          name: string;
          revoked: boolean;
          user_id: string | null;
        };
        Insert: {
          created_at?: string;
          id?: string;
          key_hash: string;
          last_used?: string;
          name: string;
          revoked?: boolean;
          user_id?: string | null;
        };
        Update: {
          created_at?: string;
          id?: string;
          key_hash?: string;
          last_used?: string;
          name?: string;
          revoked?: boolean;
          user_id?: string | null;
        };
        Relationships: [];
      };
      experiment: {
        Row: {
          created_at: string;
          description: string | null;
          hyperparams: Json[] | null;
          id: string;
          name: string;
          tags: string[] | null;
          visibility: Database["public"]["Enums"]["visibility"];
        };
        Insert: {
          created_at?: string;
          description?: string | null;
          hyperparams?: Json[] | null;
          id?: string;
          name: string;
          tags?: string[] | null;
          visibility?: Database["public"]["Enums"]["visibility"];
        };
        Update: {
          created_at?: string;
          description?: string | null;
          hyperparams?: Json[] | null;
          id?: string;
          name?: string;
          tags?: string[] | null;
          visibility?: Database["public"]["Enums"]["visibility"];
        };
        Relationships: [];
      };
      experiment_references: {
        Row: {
          created_at: string | null;
          from_experiment: string;
          id: number;
          to_experiment: string;
        };
        Insert: {
          created_at?: string | null;
          from_experiment: string;
          id?: number;
          to_experiment: string;
        };
        Update: {
          created_at?: string | null;
          from_experiment?: string;
          id?: number;
          to_experiment?: string;
        };
        Relationships: [
          {
            foreignKeyName: "experiment_references_to_experiment_fkey";
            columns: ["to_experiment"];
            isOneToOne: false;
            referencedRelation: "experiment";
            referencedColumns: ["id"];
          },
          {
            foreignKeyName: "experiment_references_to_experiment_fkey1";
            columns: ["to_experiment"];
            isOneToOne: false;
            referencedRelation: "experiment";
            referencedColumns: ["id"];
          },
        ];
      };
      metric: {
        Row: {
          created_at: string;
          experiment_id: string | null;
          id: number;
          metadata: Json | null;
          name: string;
          step: number | null;
          value: number;
        };
        Insert: {
          created_at?: string;
          experiment_id?: string | null;
          id?: number;
          metadata?: Json | null;
          name: string;
          step?: number | null;
          value: number;
        };
        Update: {
          created_at?: string;
          experiment_id?: string | null;
          id?: number;
          metadata?: Json | null;
          name?: string;
          step?: number | null;
          value?: number;
        };
        Relationships: [
          {
            foreignKeyName: "metric_experiment_id_fkey";
            columns: ["experiment_id"];
            isOneToOne: false;
            referencedRelation: "experiment";
            referencedColumns: ["id"];
          },
        ];
      };
      user_experiments: {
        Row: {
          added_at: string;
          experiment_id: string;
          role: string;
          user_id: string;
        };
        Insert: {
          added_at?: string;
          experiment_id: string;
          role?: string;
          user_id: string;
        };
        Update: {
          added_at?: string;
          experiment_id?: string;
          role?: string;
          user_id?: string;
        };
        Relationships: [
          {
            foreignKeyName: "fk_experiment";
            columns: ["experiment_id"];
            isOneToOne: false;
            referencedRelation: "experiment";
            referencedColumns: ["id"];
          },
        ];
      };
      workspace: {
        Row: {
          created_at: string;
          description: string | null;
          id: string;
          name: string;
          user_id: string;
        };
        Insert: {
          created_at?: string;
          description?: string | null;
          id?: string;
          name: string;
          user_id: string;
        };
        Update: {
          created_at?: string;
          description?: string | null;
          id?: string;
          name?: string;
          user_id?: string;
        };
        Relationships: [];
      };
      workspace_experiments: {
        Row: {
          experiment_id: string;
          workspace_id: string;
        };
        Insert: {
          experiment_id: string;
          workspace_id: string;
        };
        Update: {
          experiment_id?: string;
          workspace_id?: string;
        };
        Relationships: [
          {
            foreignKeyName: "workspace_experiments_experiment_id_fkey";
            columns: ["experiment_id"];
            isOneToOne: false;
            referencedRelation: "experiment";
            referencedColumns: ["id"];
          },
          {
            foreignKeyName: "workspace_experiments_workspace_id_fkey";
            columns: ["workspace_id"];
            isOneToOne: false;
            referencedRelation: "workspace";
            referencedColumns: ["id"];
          },
        ];
      };
    };
    Views: {
      [_ in never]: never;
    };
    Functions: {
      check_experiment_access: {
        Args: { experiment_id: string; user_id: string };
        Returns: {
          has_access: boolean;
        }[];
      };
      get_experiment_chain: {
        Args: { target_experiment_id: string };
        Returns: {
          created_at: string;
          description: string | null;
          hyperparams: Json[] | null;
          id: string;
          name: string;
          tags: string[] | null;
          visibility: Database["public"]["Enums"]["visibility"];
        }[];
      };
    };
    Enums: {
      visibility: "PUBLIC" | "PRIVATE";
    };
    CompositeTypes: {
      [_ in never]: never;
    };
  };
};

type DefaultSchema = Database[Extract<keyof Database, "public">];

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database;
  }
    ? keyof (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R;
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R;
      }
      ? R
      : never
    : never;

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database;
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I;
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I;
      }
      ? I
      : never
    : never;

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database;
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U;
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U;
      }
      ? U
      : never
    : never;

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof Database },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof Database;
  }
    ? keyof Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never;

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof Database },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof Database;
  }
    ? keyof Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends { schema: keyof Database }
  ? Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never;

export const Constants = {
  public: {
    Enums: {
      visibility: ["PUBLIC", "PRIVATE"],
    },
  },
} as const;
