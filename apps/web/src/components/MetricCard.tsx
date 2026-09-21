import { ReactNode } from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  icon?: ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  subtitle?: string;
}

export function MetricCard({ title, value, icon, trend, subtitle }: MetricCardProps) {
  return (
    <div className="bg-card border border-border rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-muted-foreground tracking-wide uppercase">{title}</h3>
        {icon && <div className="text-primary">{icon}</div>}
      </div>
      <div className="flex items-end justify-between">
        <div>
          <div className="text-3xl font-bold text-foreground">{value}</div>
          {subtitle && <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>}
        </div>
        {trend && (
          <div className={`flex items-center text-sm font-medium ${trend.isPositive ? 'text-green-500' : 'text-red-500'}`}>
            {trend.isPositive ? '↑' : '↓'}
            <span className="ml-1">{Math.abs(trend.value)}%</span>
          </div>
        )}
      </div>
    </div>
  );
}
