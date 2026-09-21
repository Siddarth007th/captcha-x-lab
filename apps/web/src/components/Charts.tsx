"use client";

import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface LineChartProps {
  data: any[];
  xKey: string;
  lines: { key: string; color: string; name?: string }[];
}

export function CustomLineChart({ data, xKey, lines }: LineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#272a30" vertical={false} />
        <XAxis 
          dataKey={xKey} 
          stroke="#94a3b8" 
          tick={{ fill: '#94a3b8' }} 
          tickLine={{ stroke: '#272a30' }}
        />
        <YAxis 
          stroke="#94a3b8" 
          tick={{ fill: '#94a3b8' }} 
          tickLine={{ stroke: '#272a30' }}
          axisLine={false}
        />
        <Tooltip 
          contentStyle={{ backgroundColor: '#15181e', borderColor: '#272a30', borderRadius: '8px', color: '#e2e8f0' }}
          itemStyle={{ color: '#e2e8f0' }}
        />
        <Legend wrapperStyle={{ paddingTop: '20px' }} />
        {lines.map((line) => (
          <Line
            key={line.key}
            type="monotone"
            dataKey={line.key}
            stroke={line.color}
            name={line.name || line.key}
            strokeWidth={3}
            dot={{ r: 4, strokeWidth: 2 }}
            activeDot={{ r: 6 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function CustomBarChart({ data, xKey, bars }: LineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#272a30" vertical={false} />
        <XAxis 
          dataKey={xKey} 
          stroke="#94a3b8" 
          tick={{ fill: '#94a3b8' }} 
          tickLine={{ stroke: '#272a30' }}
        />
        <YAxis 
          stroke="#94a3b8" 
          tick={{ fill: '#94a3b8' }} 
          tickLine={{ stroke: '#272a30' }}
          axisLine={false}
        />
        <Tooltip 
          contentStyle={{ backgroundColor: '#15181e', borderColor: '#272a30', borderRadius: '8px', color: '#e2e8f0' }}
          cursor={{ fill: '#1e2128' }}
        />
        <Legend wrapperStyle={{ paddingTop: '20px' }} />
        {bars.map((bar) => (
          <Bar
            key={bar.key}
            dataKey={bar.key}
            fill={bar.color}
            name={bar.name || bar.key}
            radius={[4, 4, 0, 0]}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
