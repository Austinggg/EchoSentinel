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
export interface UserFullProfile extends UserProfile {
  id?: string;
  hash_sec_uid?: string;
  signature?: string;
  show_favorite_list?: string;
  is_gov_media_vip?: string;
  is_mix_user?: string;
  is_star?: string;
  is_series_user?: string;
  avatar_medium?: string;
}
export interface UserCluster {
  cluster_id: string;
  avatar_list: string[];
  sec_uids: string[];
}
