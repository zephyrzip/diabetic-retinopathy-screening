import Navbar from "./Navbar"; import Sidebar from "./Sidebar";
export default function DashboardLayout({ children }) { return <><Navbar/><Sidebar/><main>{children}</main></>; }
