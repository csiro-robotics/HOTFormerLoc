interface NavigationItem {
  label: string;
  path: string;
  children?: NavigationItem[];
}
interface NavigationBarProps {
  navigationItems: NavigationItem[];
}