import Link from "next/link";
import { LayoutDashboard, Eye, Brain, Mic, LayoutTemplate, Database, Beaker } from "lucide-react";

export function Sidebar() {
  return (
    <aside className="w-64 bg-card border-r border-border h-full flex flex-col shadow-lg">
      <div className="p-6">
        <h1 className="text-xl font-bold tracking-wider text-primary">CAPTCHA-X Lab</h1>
        <p className="text-sm text-muted-foreground mt-1 font-mono uppercase">Research Platform</p>
      </div>

      <nav className="flex-1 px-4 space-y-2 mt-4">
        <SidebarLink href="/" icon={<LayoutDashboard size={20} />} label="Overview" />
        
        <div className="pt-4 pb-2">
          <p className="px-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Models</p>
        </div>
        <SidebarLink href="/vision" icon={<Eye size={20} />} label="Vision" />
        <SidebarLink href="/reasoning" icon={<Brain size={20} />} label="Reasoning" />
        <SidebarLink href="/audio" icon={<Mic size={20} />} label="Audio" />
        <SidebarLink href="/master" icon={<LayoutTemplate size={20} />} label="Master" />

        <div className="pt-4 pb-2">
          <p className="px-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Data & Runs</p>
        </div>
        <SidebarLink href="/experiments" icon={<Beaker size={20} />} label="Experiments" />
        <SidebarLink href="/datasets" icon={<Database size={20} />} label="Datasets" />
      </nav>
      
      <div className="p-4 border-t border-border">
        <div className="flex items-center space-x-3">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
          <span className="text-sm font-medium text-muted-foreground">API Connected</span>
        </div>
      </div>
    </aside>
  );
}

function SidebarLink({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) {
  return (
    <Link 
      href={href} 
      className="flex items-center space-x-3 px-3 py-2 rounded-lg text-foreground hover:bg-muted hover:text-primary transition-colors group"
    >
      <span className="text-muted-foreground group-hover:text-primary transition-colors">{icon}</span>
      <span className="font-medium">{label}</span>
    </Link>
  );
}
