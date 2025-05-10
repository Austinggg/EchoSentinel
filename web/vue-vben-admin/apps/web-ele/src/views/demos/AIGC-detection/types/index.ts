export interface EchoIdentify {
  face: string;
  body: string;
  whole: string;
}
export interface EchoIdentifyTableRow extends EchoIdentify {
  id: string;
}
export interface EchoIdentifyTableHeader {
  key: string;
  dataKey: string;
  title: string;
  width: number;
}
