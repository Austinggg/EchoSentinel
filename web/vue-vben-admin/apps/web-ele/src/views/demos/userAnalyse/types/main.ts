export interface UserProfile {
  sec_uid: string;
  nickname: string;
  gender: string;
  city: string;
  province: string;
  country: string;
  aweme_count: string;
  follower_count: string;
  following_count: string;
  total_favorited: string;
  favoriting_count: string;
  user_age: string;
  ip_location: string;
  covers: string[];
}
export interface RankInfo {
  lossValue: number;
  anomalyScore: number;
}
