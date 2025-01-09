interface NavigationItem {
  label: string;
  path: string;
  children?: NavigationItem[];
}