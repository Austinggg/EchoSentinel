import type { Column } from 'element-plus';

export interface EchoDetectionData {
  id: string;
  videoCover?: string;
  face: string;
  body: string;
  whole: string;
  filename: string;
}
export interface CellRenderProps<T = any> {
  cellData: T;
  column: Column<T>;
  columns: Column<T>[];
  columnIndex: number;
  rowData: EchoDetectionData;
  rowIndex: number;
}
export enum Alignment {
  CENTER = 'center',
  RIGHT = 'right',
}
