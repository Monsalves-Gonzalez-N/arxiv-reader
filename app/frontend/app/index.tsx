import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  StatusBar,
  ScrollView,
  Linking,
  Share,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { GestureHandlerRootView, Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, { useSharedValue, useAnimatedStyle, runOnJS } from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createClient, Session } from '@supabase/supabase-js';
import * as WebBrowser from 'expo-web-browser';
import * as ExpoLinking from 'expo-linking';

WebBrowser.maybeCompleteAuthSession();

const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL || '';
const SUPABASE_ANON_KEY = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || '';

const safeStorage = {
  getItem: (key: string) => (typeof window === 'undefined' && Platform.OS === 'web' ? Promise.resolve(null) : AsyncStorage.getItem(key)),
  setItem: (key: string, value: string) => (typeof window === 'undefined' && Platform.OS === 'web' ? Promise.resolve() : AsyncStorage.setItem(key, value)),
  removeItem: (key: string) => (typeof window === 'undefined' && Platform.OS === 'web' ? Promise.resolve() : AsyncStorage.removeItem(key)),
};

const supabase = createClient(
  SUPABASE_URL,
  SUPABASE_ANON_KEY,
  { auth: { storage: safeStorage, autoRefreshToken: true, persistSession: true, detectSessionInUrl: false } }
);

const EXPO_PUBLIC_BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || '';
const SWIPE_THRESHOLD = 50;
const COFFEE_LINK = 'https://buymeacoffee.com/Nicolasmonsalves';
const MAX_CHARS_PER_PART = 450;

interface Paper {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  link: string;
  category: string;
  comment?: string | null;
  is_recommendation?: boolean;
  is_new?: boolean;
}

const CATEGORIES = [
  { key: 'astro-ph.GA', label: 'Galaxies' },
  { key: 'astro-ph.CO', label: 'Cosmology' },
  { key: 'astro-ph.EP', label: 'Planetary' },
  { key: 'astro-ph.HE', label: 'High Energy' },
  { key: 'astro-ph.SR', label: 'Stellar' },
  { key: 'astro-ph.IM', label: 'Instrumentation' },
];

const YEARS = Array.from({ length: 36 }, (_, i) => 2025 - i);
const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DAYS = Array.from({ length: 31 }, (_, i) => i + 1);

function cleanLatex(text: string): string {
  if (!text) return text;
  return text
    .replace(/\$([^$]+)\$/g, '$1')
    .replace(/\\textbf\{([^}]+)\}/g, '$1')
    .replace(/\\textit\{([^}]+)\}/g, '$1')
    .replace(/\\emph\{([^}]+)\}/g, '$1')
    .replace(/\\text\{([^}]+)\}/g, '$1')
    .replace(/\\mathrm\{([^}]+)\}/g, '$1')
    .replace(/\\mathbf\{([^}]+)\}/g, '$1')
    .replace(/\\mathit\{([^}]+)\}/g, '$1')
    .replace(/\\sim/g, '~').replace(/\\approx/g, '≈').replace(/\\times/g, '×')
    .replace(/\\pm/g, '±').replace(/\\alpha/g, 'α').replace(/\\beta/g, 'β')
    .replace(/\\gamma/g, 'γ').replace(/\\delta/g, 'δ').replace(/\\sigma/g, 'σ')
    .replace(/\\lambda/g, 'λ').replace(/\\mu/g, 'μ').replace(/\\nu/g, 'ν')
    .replace(/\\pi/g, 'π').replace(/\\omega/g, 'ω').replace(/\\Omega/g, 'Ω')
    .replace(/\\infty/g, '∞').replace(/\\leq/g, '≤').replace(/\\geq/g, '≥')
    .replace(/\\[a-zA-Z]+\{([^}]+)\}/g, '$1').replace(/\{|\}/g, '')
    .replace(/\s+/g, ' ').trim();
}

function splitAbstract(abstract: string): string[] {
  if (!abstract) return [''];
  const sentences = abstract.match(/[^.!?]+[.!?]+/g) || [abstract];
  const parts: string[] = [];
  let cur = '';
  for (const s of sentences) {
    const t = s.trim();
    if (cur.length + t.length <= MAX_CHARS_PER_PART) cur += (cur ? ' ' : '') + t;
    else { if (cur) parts.push(cur); cur = t; }
  }
  if (cur) parts.push(cur);
  return parts.length > 0 ? parts : [''];
}

const STOP_WORDS = new Set(['the','and','for','this','that','with','from','are','was','were','been','have','has','not','but','they','their','them','its','our','which','when','than','also','into','more','such','these','those','been','will','can','may','would','could','should','using','used','show','shows','find','found','results','result','data','model','paper','study','work','two','one','new']);

function rerankPapers(allPapers: Paper[], fromIndex: number, signalPapers: Paper[]): Paper[] {
  if (signalPapers.length === 0 || fromIndex >= allPapers.length - 1) return allPapers;
  const viewed = allPapers.slice(0, fromIndex);
  const remaining = allPapers.slice(fromIndex);
  const wordFreq = new Map<string, number>();
  for (const p of signalPapers) {
    const words = `${p.title} ${p.abstract}`.toLowerCase().match(/[a-z]{5,}/g) || [];
    for (const w of words) {
      if (!STOP_WORDS.has(w)) wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
    }
  }
  const maxFreq = Math.max(...wordFreq.values(), 1);
  const scored = remaining.map(p => {
    const words = `${p.title} ${p.abstract}`.toLowerCase().match(/[a-z]{5,}/g) || [];
    let score = 0;
    for (const w of words) score += (wordFreq.get(w) || 0) / maxFreq;
    return { p, score };
  });
  const maxScore = Math.max(...scored.map(s => s.score));
  if (maxScore < 0.5) return allPapers; // not enough signal — keep original order
  scored.sort((a, b) => b.score - a.score);
  return [...viewed, ...scored.map(s => s.p)];
}

function DateSpinner({ value, min, max, onChange, label, fmt }: {
  value: number; min: number; max: number;
  onChange: (v: number) => void; label: string;
  fmt?: (v: number) => string;
}) {
  return (
    <View style={styles.spinner}>
      <Text style={styles.spinnerLabel}>{label}</Text>
      <TouchableOpacity onPress={() => onChange(value < max ? value + 1 : min)} style={styles.spinnerArrow}>
        <Ionicons name="chevron-up" size={18} color="#000" />
      </TouchableOpacity>
      <Text style={styles.spinnerValue}>{fmt ? fmt(value) : String(value).padStart(2, '0')}</Text>
      <TouchableOpacity onPress={() => onChange(value > min ? value - 1 : max)} style={styles.spinnerArrow}>
        <Ionicons name="chevron-down" size={18} color="#000" />
      </TouchableOpacity>
    </View>
  );
}

export default function Index() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set([]));
  const [likedPapers, setLikedPapers] = useState<Set<string>>(new Set());
  const [likedPapersList, setLikedPapersList] = useState<any[]>([]);

  // Date filter state
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [selectedMonth, setSelectedMonth] = useState<number | null>(null);
  const [customFrom, setCustomFrom] = useState<string | null>(null); // YYYYMMDD
  const [customTo, setCustomTo] = useState<string | null>(null);     // YYYYMMDD

  // Date picker UI
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [datePickerMode, setDatePickerMode] = useState<'months' | 'custom'>('months');
  const [pickerYear, setPickerYear] = useState(new Date().getFullYear());
  const now = new Date();
  const [fromDay, setFromDay] = useState(1);
  const [fromMonth, setFromMonth] = useState(1);
  const [fromYear, setFromYear] = useState(now.getFullYear());
  const [toDay, setToDay] = useState(now.getDate());
  const [toMonth, setToMonth] = useState(now.getMonth() + 1);
  const [toYear, setToYear] = useState(now.getFullYear());

  const [currentView, setCurrentView] = useState<'feed' | 'likes' | 'settings'>('feed');
  const [abstractPart, setAbstractPart] = useState(0);
  const [showCategoryPicker, setShowCategoryPicker] = useState(false);
  const [draftCategories, setDraftCategories] = useState<Set<string>>(new Set());
  const [session, setSession] = useState<Session | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [showOnboarding, setShowOnboarding] = useState(false);

  const olderStartRef = useRef(0);
  const dwellStartRef = useRef<number>(Date.now());
  const engagedIdsRef = useRef<Set<string>>(new Set());
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);

  const currentPaper = papers[currentIndex];
  const abstractParts = useMemo(() =>
    currentPaper ? splitAbstract(cleanLatex(currentPaper.abstract)) : [''],
    [currentPaper?.abstract]
  );
  const totalParts = abstractParts.length + 1;

  // Date label for header button
  const isToday = !selectedYear && !customFrom;
  const todayStr = new Date().toISOString().split('T')[0];
  const papersDate = isToday && papers.length > 0 ? papers[0].published : null;
  const showOldPapersBanner = isToday && papersDate !== null && papersDate < todayStr;
  const dateLabel = customFrom && customTo
    ? `${customFrom.slice(6)}/${MONTH_NAMES[parseInt(customFrom.slice(4,6))-1]}–${customTo.slice(6)}/${MONTH_NAMES[parseInt(customTo.slice(4,6))-1]}`
    : selectedYear
      ? `${MONTH_NAMES[(selectedMonth ?? 1) - 1]} ${selectedYear}`
      : 'Today';

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setAuthLoading(false);
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });
    loadSavedCategories();
    AsyncStorage.getItem('onboarding_seen').then(val => { if (!val) setShowOnboarding(true); });
    return () => subscription.unsubscribe();
  }, []);

  const signInWithGoogle = async () => {
    const redirectTo = Platform.OS === 'web'
      ? `${window.location.origin}/auth/callback`
      : ExpoLinking.createURL('auth/callback');
    await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo },
    });
  };

  const signOut = async () => {
    await supabase.auth.signOut();
  };

  const getAuthHeaders = async (): Promise<Record<string, string>> => {
    const { data: { session: s } } = await supabase.auth.getSession();
    if (s?.access_token) return { 'Authorization': `Bearer ${s.access_token}` };
    const id = await AsyncStorage.getItem('device_id');
    return id ? { 'X-Device-ID': id } : {};
  };

  const loadSavedCategories = async () => {
    try {
      const saved = await AsyncStorage.getItem('selected_categories');
      setSelectedCategories(saved ? new Set(JSON.parse(saved)) : new Set(['astro-ph.GA', 'astro-ph.CO']));
    } catch {
      setSelectedCategories(new Set(['astro-ph.GA', 'astro-ph.CO']));
    }
  };

  const fetchPapers = useCallback(async (
    categories: Set<string>,
    year: number | null,
    month: number | null,
    dateFrom: string | null,
    dateTo: string | null,
    start: number = 0,
    append: boolean = false,
  ) => {
    try {
      if (!append) setLoading(true);
      else setLoadingMore(true);

      const headers = await getAuthHeaders();
      const category = Array.from(categories).join(',');

      let url: string;
      if (dateFrom && dateTo) {
        url = `${EXPO_PUBLIC_BACKEND_URL}/api/papers?category=${category}&date_from=${dateFrom}&date_to=${dateTo}&start=${start}&max_results=25`;
      } else if (year) {
        url = `${EXPO_PUBLIC_BACKEND_URL}/api/papers?category=${category}&year=${year}&month=${month ?? 1}&start=${start}&max_results=25`;
      } else {
        url = `${EXPO_PUBLIC_BACKEND_URL}/api/papers/feed?category=${category}`;
      }

      const response = await fetch(url, { headers });
      const data = await response.json();
      const fetched: Paper[] = data.papers || [];

      if (append) {
        setPapers(prev => {
          const ids = new Set(prev.map(p => p.id));
          const newOnes = fetched.filter(p => !ids.has(p.id));
          olderStartRef.current += newOnes.length;
          return [...prev, ...newOnes];
        });
      } else {
        setPapers(fetched);
        setAbstractPart(0);
        olderStartRef.current = fetched.length;
        dwellStartRef.current = Date.now();
        engagedIdsRef.current = new Set();
        // Restore saved position for today's feed
        if (!year && !dateFrom) {
          try {
            const saved = await AsyncStorage.getItem('feed_position');
            if (saved) {
              const { date, paperId } = JSON.parse(saved);
              const today = new Date().toISOString().split('T')[0];
              if (date === today) {
                const idx = fetched.findIndex(p => p.id === paperId);
                if (idx > 0) { setCurrentIndex(idx); return; }
              }
            }
          } catch {}
        }
        setCurrentIndex(0);
      }
      setHasMore(data.has_more ?? false);
    } catch (error) {
      console.error('Error fetching papers:', error);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, []);

  const fetchLikedPapers = useCallback(async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${EXPO_PUBLIC_BACKEND_URL}/api/likes`, { headers });
      const data = await response.json();
      setLikedPapers(new Set(data.map((p: any) => p.paper_id)));
      setLikedPapersList(data);
    } catch (error) {
      console.error('Error fetching liked papers:', error);
    }
  }, []);

  useEffect(() => {
    if (selectedCategories.size > 0 && session) {
      olderStartRef.current = 0;
      fetchPapers(selectedCategories, selectedYear, selectedMonth, customFrom, customTo);
      fetchLikedPapers();
    }
  }, [session]); // eslint-disable-line react-hooks/exhaustive-deps
  // Note: selectedCategories intentionally omitted — category changes trigger fetches
  // explicitly via the Apply button to avoid a fetch on every checkbox tap.

  useEffect(() => {
    if (selectedCategories.size > 0) {
      olderStartRef.current = 0;
      fetchPapers(selectedCategories, selectedYear, selectedMonth, customFrom, customTo);
    }
  }, [selectedYear, selectedMonth, customFrom, customTo]);

  // Persist position (today mode only) so user resumes where they left off
  useEffect(() => {
    if (!selectedYear && !customFrom && papers.length > 0 && currentIndex < papers.length) {
      const today = new Date().toISOString().split('T')[0];
      const paperId = papers[currentIndex]?.id;
      if (paperId) AsyncStorage.setItem('feed_position', JSON.stringify({ date: today, paperId }));
    }
  }, [currentIndex]);

  const openCategoryPicker = () => {
    setDraftCategories(new Set(selectedCategories));
    setShowCategoryPicker(true);
  };

  const toggleCategory = (key: string) => {
    setDraftCategories(prev => {
      const s = new Set(prev);
      if (s.has(key)) { if (s.size > 1) s.delete(key); } else s.add(key);
      return s;
    });
  };

  const applyCategories = () => {
    setSelectedCategories(draftCategories);
    AsyncStorage.setItem('selected_categories', JSON.stringify(Array.from(draftCategories)));
    setShowCategoryPicker(false);
    fetchPapers(draftCategories, selectedYear, selectedMonth, customFrom, customTo);
  };

  const toggleLike = async (paper: Paper) => {
    const isLiked = likedPapers.has(paper.id);
    const headers: any = { ...(await getAuthHeaders()), 'Content-Type': 'application/json' };
    try {
      if (isLiked) {
        await fetch(`${EXPO_PUBLIC_BACKEND_URL}/api/likes/${paper.id}`, { method: 'DELETE', headers });
        setLikedPapers(prev => { const s = new Set(prev); s.delete(paper.id); return s; });
      } else {
        await fetch(`${EXPO_PUBLIC_BACKEND_URL}/api/likes`, {
          method: 'POST', headers,
          body: JSON.stringify({ paper_id: paper.id, title: paper.title, abstract: paper.abstract, authors: paper.authors, published: paper.published, link: paper.link, category: paper.category }),
        });
        setLikedPapers(prev => new Set(prev).add(paper.id));
      }
      fetchLikedPapers();
    } catch (error) { console.error('Error toggling like:', error); }
  };

  const goToNext = useCallback(() => {
    setAbstractPart(0);
    const next = currentIndex + 1;

    // Track dwell time — >8s counts as implicit engagement signal
    const dwell = Date.now() - dwellStartRef.current;
    dwellStartRef.current = Date.now();
    if (dwell > 8000 && papers[currentIndex]) {
      engagedIdsRef.current.add(papers[currentIndex].id);
    }

    if (next < papers.length) {
      // Re-rank remaining papers based on likes + engagement
      const signalPapers = papers.filter(p => likedPapers.has(p.id) || engagedIdsRef.current.has(p.id));
      if (signalPapers.length > 0) {
        setPapers(prev => rerankPapers(prev, next, signalPapers));
      }
      setCurrentIndex(next);
      if (hasMore && papers.length - next < 5 && !loadingMore) {
        fetchPapers(selectedCategories, selectedYear, selectedMonth, customFrom, customTo, olderStartRef.current, true);
      }
    } else if (hasMore && !loadingMore) {
      fetchPapers(selectedCategories, selectedYear, selectedMonth, customFrom, customTo, olderStartRef.current, true);
    } else {
      setCurrentIndex(papers.length);
    }
  }, [currentIndex, papers, likedPapers, hasMore, loadingMore, selectedCategories, selectedYear, selectedMonth, customFrom, customTo, fetchPapers]);

  const goToPrevious = useCallback(() => {
    if (currentIndex > 0) { setCurrentIndex(prev => prev - 1); setAbstractPart(0); }
  }, [currentIndex]);

  const nextAbstractPart = useCallback(() => {
    if (abstractPart < abstractParts.length) setAbstractPart(prev => prev + 1);
  }, [abstractPart, abstractParts.length]);

  const prevAbstractPart = useCallback(() => {
    if (abstractPart > 0) setAbstractPart(prev => prev - 1);
  }, [abstractPart]);

  useEffect(() => {
    if (Platform.OS !== 'web') return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') goToNext();
      else if (e.key === 'ArrowUp') goToPrevious();
      else if (e.key === 'ArrowRight') nextAbstractPart();
      else if (e.key === 'ArrowLeft') prevAbstractPart();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [goToNext, goToPrevious, nextAbstractPart, prevAbstractPart]);

  const panGesture = Gesture.Pan()
    .minDistance(10)
    .onUpdate(e => { translateX.value = e.translationX; translateY.value = e.translationY; })
    .onEnd(e => {
      const { translationX: tx, translationY: ty, velocityX: vx, velocityY: vy } = e;
      if (Math.abs(ty) > Math.abs(tx) * 0.8) {
        if (ty < -SWIPE_THRESHOLD || vy < -500) runOnJS(goToNext)();
        else if (ty > SWIPE_THRESHOLD || vy > 500) runOnJS(goToPrevious)();
      } else if (Math.abs(tx) > SWIPE_THRESHOLD || Math.abs(vx) > 500) {
        if (tx < -SWIPE_THRESHOLD || vx < -500) runOnJS(nextAbstractPart)();
        else if (tx > SWIPE_THRESHOLD || vx > 500) runOnJS(prevAbstractPart)();
      }
      translateX.value = 0; translateY.value = 0;
    });

  const animatedCardStyle = useAnimatedStyle(() => ({ transform: [] }));

  const applyCustomRange = () => {
    const from = `${fromYear}${String(fromMonth).padStart(2,'0')}${String(fromDay).padStart(2,'0')}`;
    const to = `${toYear}${String(toMonth).padStart(2,'0')}${String(toDay).padStart(2,'0')}`;
    setCustomFrom(from);
    setCustomTo(to);
    setSelectedYear(null);
    setSelectedMonth(null);
    setShowDatePicker(false);
    setDatePickerMode('months');
  };

  const exportToCSV = async () => {
    if (likedPapersList.length === 0) return;
    const h = 'Title,Authors,Published,Link\n';
    const rows = likedPapersList.map(p => `"${p.title.replace(/"/g, '""')}","${p.authors.join('; ')}",${p.published},${p.link}`).join('\n');
    const csv = h + rows;
    if (Platform.OS === 'web') {
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'arxiv_papers.csv';
      a.click();
      URL.revokeObjectURL(url);
    } else {
      await Share.share({ message: csv, title: 'My arXiv Papers' });
    }
  };

  const openCoffee = () => Linking.openURL(COFFEE_LINK);
  const openPaper = () => currentPaper && Linking.openURL(currentPaper.link);

  const dismissOnboarding = useCallback(() => {
    setShowOnboarding(false);
    AsyncStorage.setItem('onboarding_seen', '1');
  }, []);

  useEffect(() => {
    if (!showOnboarding) return;
    const t = setTimeout(dismissOnboarding, 18000);
    return () => clearTimeout(t);
  }, [showOnboarding, dismissOnboarding]);

  const allPapersSeen = !loading && papers.length > 0 && currentIndex >= papers.length;
  const noPapersFound = !loading && papers.length === 0;

  if (authLoading) {
    return (
      <View style={styles.loadingContainer}>
        <StatusBar barStyle="dark-content" />
        <ActivityIndicator size="large" color="#000" />
      </View>
    );
  }

  if (!session) {
    return (
      <View style={styles.loginContainer}>
        <StatusBar barStyle="dark-content" />
        <Text style={styles.loginTitle}>arXiv Reader</Text>
        <Text style={styles.loginSubtitle}>Daily astrophysics papers, personalized for you</Text>
        <TouchableOpacity style={styles.googleBtn} onPress={signInWithGoogle}>
          <Ionicons name="logo-google" size={20} color="#fff" />
          <Text style={styles.googleBtnText}>Continue with Google</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => setSession({ user: { id: 'guest' } } as any)}>
          <Text style={styles.guestBtn}>Continue as guest</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (loading && papers.length === 0) {
    return (
      <View style={styles.loadingContainer}>
        <StatusBar barStyle="dark-content" />
        <ActivityIndicator size="large" color="#000" />
        <Text style={styles.loadingTitle}>Fetching papers</Text>
        <Text style={styles.loadingText}>Retrieving today's arXiv submissions…</Text>
      </View>
    );
  }

  // Likes View
  if (currentView === 'likes') {
    return (
      <GestureHandlerRootView style={styles.container}>
        <StatusBar barStyle="dark-content" />
        <SafeAreaView style={styles.safeArea}>
          <View style={styles.header}>
            <TouchableOpacity onPress={() => setCurrentView('feed')}>
              <Ionicons name="arrow-back" size={24} color="#000" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>Saved Papers</Text>
            <TouchableOpacity onPress={exportToCSV}>
              <Ionicons name="download-outline" size={24} color="#000" />
            </TouchableOpacity>
          </View>
          <ScrollView style={styles.likesList}>
            {likedPapersList.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="heart-outline" size={48} color="#ccc" />
                <Text style={styles.emptyText}>No saved papers</Text>
              </View>
            ) : likedPapersList.map((paper, idx) => (
              <TouchableOpacity key={paper.paper_id || idx} style={styles.likedCard} onPress={() => Linking.openURL(paper.link)}>
                <View style={styles.likedContent}>
                  <Text style={styles.likedTitle} numberOfLines={2}>{paper.title}</Text>
                  <Text style={styles.likedAuthors} numberOfLines={1}>{paper.authors.slice(0, 2).join(', ')}</Text>
                  <Text style={styles.likedDate}>{paper.published}</Text>
                </View>
                <TouchableOpacity onPress={() => toggleLike(paper)}>
                  <Ionicons name="heart" size={20} color="#000" />
                </TouchableOpacity>
              </TouchableOpacity>
            ))}
          </ScrollView>
          {likedPapersList.length > 0 && (
            <TouchableOpacity style={styles.exportBtn} onPress={exportToCSV}>
              <Text style={styles.exportBtnText}>Export CSV</Text>
            </TouchableOpacity>
          )}
        </SafeAreaView>
      </GestureHandlerRootView>
    );
  }

  // Settings View
  if (currentView === 'settings') {
    return (
      <GestureHandlerRootView style={styles.container}>
        <StatusBar barStyle="dark-content" />
        <SafeAreaView style={styles.safeArea}>
          <View style={styles.header}>
            <TouchableOpacity onPress={() => setCurrentView('feed')}>
              <Ionicons name="arrow-back" size={24} color="#000" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>Settings</Text>
            <View style={{ width: 24 }} />
          </View>
          <ScrollView style={styles.settingsContent}>
            <View style={styles.divider} />
            <TouchableOpacity style={styles.supportCard} onPress={openCoffee}>
              <View style={styles.supportCardContent}>
                <Ionicons name="heart" size={20} color="#fff" />
                <View style={{ flex: 1 }}>
                  <Text style={styles.supportCardTitle}>Support this project</Text>
                  <Text style={styles.supportCardSub}>Help keep the app free & running</Text>
                </View>
                <Ionicons name="chevron-forward" size={18} color="rgba(255,255,255,0.6)" />
              </View>
            </TouchableOpacity>
            <TouchableOpacity style={styles.settingsItem} onPress={() => Linking.openURL('https://arxiv.org')}>
              <Ionicons name="globe-outline" size={24} color="#000" />
              <View style={styles.settingsItemContent}>
                <Text style={styles.settingsItemTitle}>arXiv.org</Text>
                <Text style={styles.settingsItemSub}>Data source</Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#ccc" />
            </TouchableOpacity>
            <View style={styles.divider} />
            <View style={styles.settingsItem}>
              <Ionicons name="person-outline" size={24} color="#000" />
              <View style={styles.settingsItemContent}>
                <Text style={styles.settingsItemTitle}>{session?.user?.email}</Text>
                <Text style={styles.settingsItemSub}>Signed in with Google</Text>
              </View>
            </View>
            <TouchableOpacity style={styles.signOutBtn} onPress={signOut}>
              <Text style={styles.signOutBtnText}>Sign out</Text>
            </TouchableOpacity>
          </ScrollView>
        </SafeAreaView>
      </GestureHandlerRootView>
    );
  }

  // Main Feed View
  return (
    <GestureHandlerRootView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <SafeAreaView style={styles.safeArea}>

        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.categoryBtn} onPress={openCategoryPicker}>
            <Text style={styles.categoryBtnText}>
              {selectedCategories.size === CATEGORIES.length ? 'All' : `${selectedCategories.size} selected`}
            </Text>
            <Ionicons name="chevron-down" size={16} color="#000" />
          </TouchableOpacity>
          <View style={styles.headerRight}>
            <TouchableOpacity style={[styles.dateBtn, !isToday && styles.dateBtnActive]} onPress={() => { setDatePickerMode('months'); setShowDatePicker(true); }}>
              <Ionicons name="calendar-outline" size={14} color={isToday ? '#000' : '#fff'} />
              <Text style={[styles.dateBtnText, !isToday && styles.dateBtnTextActive]}>{dateLabel}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.iconBtn} onPress={() => { fetchLikedPapers(); setCurrentView('likes'); }}>
              <Ionicons name="heart" size={22} color="#000" />
              {likedPapers.size > 0 && (
                <View style={styles.badge}><Text style={styles.badgeText}>{likedPapers.size}</Text></View>
              )}
            </TouchableOpacity>
            <TouchableOpacity style={styles.iconBtn} onPress={() => setCurrentView('settings')}>
              <Ionicons name="person-outline" size={22} color="#000" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Old papers banner */}
        {showOldPapersBanner && (
          <View style={styles.oldPapersBanner}>
            <Ionicons name="information-circle-outline" size={14} color="#666" />
            <Text style={styles.oldPapersBannerText}>
              No new papers today — showing {papersDate ? new Date(papersDate + 'T12:00:00').toLocaleDateString('en-GB', { weekday: 'long', day: '2-digit', month: 'long', year: 'numeric' }) : ''}
            </Text>
          </View>
        )}

        {/* Card Area */}
        <GestureDetector gesture={panGesture}>
          <Animated.View style={[styles.cardArea, animatedCardStyle]}>
            {allPapersSeen ? (
              <View style={styles.emptyCard}>
                <Ionicons name="checkmark-circle-outline" size={52} color="#ccc" />
                <Text style={styles.emptyCardTitle}>
                  {isToday ? "You've seen all of today's papers!" : 'No more papers'}
                </Text>
                {isToday && (
                  <Text style={styles.emptyCardSubtext}>
                    Explore papers from a specific month or date range to discover more
                  </Text>
                )}
                <TouchableOpacity style={styles.browseBtn} onPress={() => { setDatePickerMode('months'); setShowDatePicker(true); }}>
                  <Ionicons name="calendar-outline" size={18} color="#fff" />
                  <Text style={styles.browseBtnText}>Browse by date</Text>
                </TouchableOpacity>
                {!isToday && hasMore && (
                  <TouchableOpacity style={[styles.browseBtn, { marginTop: 10, backgroundColor: '#666' }]}
                    onPress={() => fetchPapers(selectedCategories, selectedYear, selectedMonth, customFrom, customTo, olderStartRef.current, true)}>
                    <Text style={styles.browseBtnText}>Load more</Text>
                  </TouchableOpacity>
                )}
              </View>
            ) : noPapersFound ? (
              <View style={styles.emptyCard}>
                <Ionicons name="search-outline" size={48} color="#ccc" />
                <Text style={styles.emptyCardTitle}>No papers found</Text>
                <Text style={styles.emptyCardSubtext}>Try selecting different categories or a different date</Text>
              </View>
            ) : currentPaper ? (
              <View style={styles.card}>
                {currentPaper.is_new && (
                  <View style={styles.newBadge}><Text style={styles.newBadgeText}>NEW</Text></View>
                )}
                <View style={styles.cardContent}>
                  {abstractPart === 0 ? (
                    <>
                      <Text style={styles.paperTitle}>{cleanLatex(currentPaper.title)}</Text>
                      {currentPaper.authors?.length > 0 && (
                        <Text style={styles.paperAuthor}>
                          {currentPaper.authors[0]}{currentPaper.authors.length > 1 ? ` +${currentPaper.authors.length - 1}` : ''}
                        </Text>
                      )}
                      {currentPaper.comment && (
                        <View style={styles.commentBadge}>
                          <Text style={styles.commentText} numberOfLines={2}>{currentPaper.comment}</Text>
                        </View>
                      )}
                      <Text style={styles.paperDate}>{currentPaper.published}</Text>
                      <TouchableOpacity style={styles.linkBtn} onPress={openPaper}>
                        <Ionicons name="open-outline" size={16} color="#666" />
                        <Text style={styles.linkBtnText}>Open in arXiv</Text>
                      </TouchableOpacity>
                    </>
                  ) : (
                    <View style={styles.abstractContainer}>
                      <Text style={styles.abstractPartLabel}>Part {abstractPart}/{abstractParts.length}</Text>
                      <Text style={styles.abstractText} numberOfLines={12}>{abstractParts[abstractPart - 1] || ''}</Text>
                    </View>
                  )}
                </View>
                <View style={styles.dots}>
                  {Array.from({ length: totalParts }, (_, i) => (
                    <View key={i} style={[styles.dot, abstractPart === i && styles.dotActive]} />
                  ))}
                </View>
                <Text style={styles.counter}>{currentIndex + 1} / {papers.length}{hasMore ? '+' : ''}</Text>
              </View>
            ) : null}
          </Animated.View>
        </GestureDetector>

        {/* Bottom Bar */}
        <View style={styles.bottomBar}>
          <Text style={styles.hintText}>
            {abstractPart === 0 ? 'Swipe ↑↓ to navigate · ← for abstract' : 'Swipe → back · ↑↓ to change paper'}
          </Text>
          <TouchableOpacity style={styles.likeBtn} onPress={() => currentPaper && toggleLike(currentPaper)}>
            <Ionicons
              name={currentPaper && likedPapers.has(currentPaper.id) ? 'heart' : 'heart-outline'}
              size={28}
              color={currentPaper && likedPapers.has(currentPaper.id) ? '#000' : '#666'}
            />
          </TouchableOpacity>
        </View>

        {/* Category Modal */}
        {showCategoryPicker && (
          <TouchableOpacity style={styles.modalOverlay} activeOpacity={1} onPress={applyCategories}>
            <TouchableOpacity activeOpacity={1} onPress={() => {}}>
              <View style={styles.categoryModal}>
                <Text style={styles.categoryModalTitle}>Select Categories</Text>
                <Text style={styles.categoryModalSub}>Tap to select multiple</Text>
                <View style={styles.categoryGrid}>
                  {CATEGORIES.map(cat => (
                    <TouchableOpacity key={cat.key}
                      style={[styles.categoryCheckbox, draftCategories.has(cat.key) && styles.categoryCheckboxActive]}
                      onPress={() => toggleCategory(cat.key)}>
                      <View style={[styles.checkbox, draftCategories.has(cat.key) && styles.checkboxActive]}>
                        {draftCategories.has(cat.key) && <Ionicons name="checkmark" size={14} color="#fff" />}
                      </View>
                      <Text style={[styles.categoryCheckboxText, draftCategories.has(cat.key) && styles.categoryCheckboxTextActive]}>{cat.label}</Text>
                    </TouchableOpacity>
                  ))}
                </View>
                <TouchableOpacity style={styles.applyBtn} onPress={applyCategories}>
                  <Text style={styles.applyBtnText}>Apply</Text>
                </TouchableOpacity>
              </View>
            </TouchableOpacity>
          </TouchableOpacity>
        )}

        {/* Date Picker Modal */}
        {showDatePicker && (
          <TouchableOpacity style={styles.modalOverlay} activeOpacity={1} onPress={() => { setShowDatePicker(false); setDatePickerMode('months'); }}>
            <TouchableOpacity activeOpacity={1} onPress={() => {}}>
              <View style={styles.dateModal}>

                {datePickerMode === 'months' ? (
                  <>
                    <Text style={styles.dateModalTitle}>Browse by Date</Text>

                    {/* Today */}
                    <TouchableOpacity
                      style={[styles.todayOption, isToday && styles.todayOptionActive]}
                      onPress={() => { setSelectedYear(null); setSelectedMonth(null); setCustomFrom(null); setCustomTo(null); setShowDatePicker(false); }}>
                      <Text style={[styles.todayOptionText, isToday && styles.todayOptionTextActive]}>Today</Text>
                    </TouchableOpacity>

                    {/* Year nav */}
                    <View style={styles.yearNav}>
                      <TouchableOpacity onPress={() => setPickerYear(y => Math.max(1990, y - 1))} style={styles.yearNavArrow}>
                        <Ionicons name="chevron-back" size={20} color="#000" />
                      </TouchableOpacity>
                      <Text style={styles.yearNavText}>{pickerYear}</Text>
                      <TouchableOpacity onPress={() => setPickerYear(y => Math.min(now.getFullYear(), y + 1))} style={styles.yearNavArrow}>
                        <Ionicons name="chevron-forward" size={20} color="#000" />
                      </TouchableOpacity>
                    </View>

                    {/* Month grid */}
                    <View style={styles.monthGrid}>
                      {MONTH_NAMES.map((m, i) => {
                        const active = selectedYear === pickerYear && selectedMonth === i + 1 && !customFrom;
                        return (
                          <TouchableOpacity key={i} style={[styles.monthCell, active && styles.monthCellActive]}
                            onPress={() => { setSelectedYear(pickerYear); setSelectedMonth(i + 1); setCustomFrom(null); setCustomTo(null); setShowDatePicker(false); }}>
                            <Text style={[styles.monthCellText, active && styles.monthCellTextActive]}>{m}</Text>
                          </TouchableOpacity>
                        );
                      })}
                    </View>

                    {/* Custom range */}
                    <TouchableOpacity style={[styles.customRangeBtn, customFrom && styles.customRangeBtnActive]} onPress={() => setDatePickerMode('custom')}>
                      <Ionicons name="calendar-outline" size={15} color={customFrom ? '#fff' : '#666'} />
                      <Text style={[styles.customRangeBtnText, customFrom && styles.customRangeBtnTextActive]}>
                        {customFrom ? `${customFrom.slice(6)}/${MONTH_NAMES[parseInt(customFrom.slice(4,6))-1]}/${customFrom.slice(0,4)} – ${customTo!.slice(6)}/${MONTH_NAMES[parseInt(customTo!.slice(4,6))-1]}/${customTo!.slice(0,4)}` : 'Custom range'}
                      </Text>
                    </TouchableOpacity>
                  </>
                ) : (
                  <>
                    <TouchableOpacity style={styles.backRow} onPress={() => setDatePickerMode('months')}>
                      <Ionicons name="arrow-back" size={20} color="#000" />
                      <Text style={styles.backRowText}>Custom range</Text>
                    </TouchableOpacity>

                    <Text style={styles.rangeLabel}>From</Text>
                    <View style={styles.dateRow}>
                      <DateSpinner value={fromDay} min={1} max={31} onChange={setFromDay} label="Day" />
                      <DateSpinner value={fromMonth} min={1} max={12} onChange={setFromMonth} label="Month" fmt={v => MONTH_NAMES[v-1]} />
                      <DateSpinner value={fromYear} min={1990} max={now.getFullYear()} onChange={setFromYear} label="Year" fmt={v => String(v)} />
                    </View>

                    <Text style={[styles.rangeLabel, { marginTop: 16 }]}>To</Text>
                    <View style={styles.dateRow}>
                      <DateSpinner value={toDay} min={1} max={31} onChange={setToDay} label="Day" />
                      <DateSpinner value={toMonth} min={1} max={12} onChange={setToMonth} label="Month" fmt={v => MONTH_NAMES[v-1]} />
                      <DateSpinner value={toYear} min={1990} max={now.getFullYear()} onChange={setToYear} label="Year" fmt={v => String(v)} />
                    </View>

                    <TouchableOpacity style={[styles.applyBtn, { marginTop: 20 }]} onPress={applyCustomRange}>
                      <Text style={styles.applyBtnText}>Apply</Text>
                    </TouchableOpacity>
                  </>
                )}
              </View>
            </TouchableOpacity>
          </TouchableOpacity>
        )}

        {loadingMore && (
          <View style={styles.loadingMoreOverlay}>
            <ActivityIndicator size="small" color="#000" />
          </View>
        )}

        {/* First-time onboarding hint */}
        {showOnboarding && (
          <TouchableOpacity style={styles.onboardingOverlay} activeOpacity={1} onPress={dismissOnboarding}>
            <View style={styles.onboardingCard}>
              <View style={styles.onboardingCategoryHint}>
                <Ionicons name="options-outline" size={16} color="#000" />
                <Text style={styles.onboardingCategoryText}>Select your research areas ↖</Text>
              </View>
              <View style={styles.onboardingRow}>
                <Ionicons name="swap-vertical-outline" size={16} color="rgba(255,255,255,0.9)" />
                <Text style={styles.onboardingText}>
                  {Platform.OS === 'web' ? '↑ ↓   Browse papers' : 'Swipe up / down   Browse papers'}
                </Text>
              </View>
              <View style={styles.onboardingRow}>
                <Ionicons name="swap-horizontal-outline" size={16} color="rgba(255,255,255,0.9)" />
                <Text style={styles.onboardingText}>
                  {Platform.OS === 'web' ? '← →   Read abstract' : 'Swipe left / right   Read abstract'}
                </Text>
              </View>
              <View style={styles.onboardingRow}>
                <Ionicons name="heart-outline" size={16} color="rgba(255,255,255,0.9)" />
                <Text style={styles.onboardingText}>Heart to save papers</Text>
              </View>
            </View>
          </TouchableOpacity>
        )}
      </SafeAreaView>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  safeArea: { flex: 1 },
  loadingContainer: { flex: 1, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  loadingTitle: { marginTop: 16, color: '#000', fontSize: 17, fontWeight: '600' },
  loadingText: { marginTop: 6, color: '#999', fontSize: 14 },

  // Header
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#f0f0f0' },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  categoryBtn: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  categoryBtnText: { fontSize: 17, fontWeight: '600', color: '#000' },
  iconBtn: { position: 'relative', padding: 4 },
  dateBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 12, borderWidth: 1, borderColor: '#ddd' },
  dateBtnActive: { backgroundColor: '#000', borderColor: '#000' },
  dateBtnText: { fontSize: 13, fontWeight: '600', color: '#000' },
  dateBtnTextActive: { color: '#fff' },
  badge: { position: 'absolute', top: -2, right: -2, backgroundColor: '#000', borderRadius: 8, minWidth: 16, height: 16, justifyContent: 'center', alignItems: 'center' },
  badgeText: { color: '#fff', fontSize: 10, fontWeight: '600' },
  oldPapersBanner: { flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 16, paddingVertical: 6, backgroundColor: '#f5f5f5' },
  oldPapersBannerText: { fontSize: 12, color: '#666' },

  // Card
  cardArea: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  card: { width: '100%', height: '100%', backgroundColor: '#fafafa', borderRadius: 16, padding: 24, justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: '#e0e0e0', overflow: 'hidden' },
  emptyCard: { width: '100%', height: '100%', backgroundColor: '#fafafa', borderRadius: 16, padding: 32, justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: '#e0e0e0' },
  emptyCardTitle: { marginTop: 16, color: '#555', fontSize: 18, fontWeight: '700', textAlign: 'center' },
  emptyCardSubtext: { marginTop: 10, color: '#999', fontSize: 14, textAlign: 'center', lineHeight: 20, paddingHorizontal: 8 },
  browseBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 24, paddingVertical: 12, paddingHorizontal: 24, backgroundColor: '#000', borderRadius: 24 },
  browseBtnText: { color: '#fff', fontSize: 15, fontWeight: '600' },
  newBadge: { position: 'absolute', top: 16, left: 16, backgroundColor: '#000', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4 },
  newBadgeText: { color: '#fff', fontSize: 10, fontWeight: '700' },
  cardContent: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 20, width: '100%' },
  paperTitle: { fontSize: 20, fontWeight: '700', color: '#000', textAlign: 'center', lineHeight: 28 },
  paperAuthor: { marginTop: 6, fontSize: 13, color: '#666', fontStyle: 'italic' },
  paperDate: { marginTop: 16, fontSize: 14, color: '#999' },
  commentBadge: { marginTop: 12, backgroundColor: '#f0f0f0', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 6, alignSelf: 'center' },
  commentText: { fontSize: 12, color: '#666', textAlign: 'center', fontStyle: 'italic' },
  linkBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 16, paddingVertical: 8, paddingHorizontal: 16, backgroundColor: '#f5f5f5', borderRadius: 20 },
  linkBtnText: { fontSize: 13, color: '#666' },
  abstractPartLabel: { fontSize: 12, color: '#999', marginBottom: 12, fontWeight: '500' },
  abstractText: { fontSize: 15, color: '#333', textAlign: 'center', lineHeight: 22 },
  abstractContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', width: '100%', paddingHorizontal: 8 },
  dots: { flexDirection: 'row', gap: 6, marginTop: 16, flexWrap: 'wrap', justifyContent: 'center' },
  dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#ddd' },
  dotActive: { backgroundColor: '#000' },
  counter: { position: 'absolute', bottom: 16, right: 16, fontSize: 12, color: '#bbb' },

  // Bottom bar
  bottomBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 24, paddingVertical: 16, borderTopWidth: 1, borderTopColor: '#f0f0f0' },
  hintText: { fontSize: 12, color: '#bbb', flex: 1 },
  likeBtn: { width: 48, height: 48, borderRadius: 24, backgroundColor: '#f5f5f5', justifyContent: 'center', alignItems: 'center' },
  loadingMoreOverlay: { position: 'absolute', bottom: 90, right: 20 },

  // Modals
  modalOverlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', alignItems: 'center' },

  // Category modal
  categoryModal: { backgroundColor: '#fff', borderRadius: 16, padding: 24, width: 320 },
  categoryModalTitle: { fontSize: 18, fontWeight: '700', color: '#000', textAlign: 'center' },
  categoryModalSub: { fontSize: 13, color: '#999', textAlign: 'center', marginTop: 4, marginBottom: 20 },
  categoryGrid: { gap: 10 },
  categoryCheckbox: { flexDirection: 'row', alignItems: 'center', paddingVertical: 12, paddingHorizontal: 12, borderRadius: 8, borderWidth: 1, borderColor: '#e0e0e0', gap: 12 },
  categoryCheckboxActive: { borderColor: '#000', backgroundColor: '#fafafa' },
  checkbox: { width: 22, height: 22, borderRadius: 4, borderWidth: 2, borderColor: '#ccc', justifyContent: 'center', alignItems: 'center' },
  checkboxActive: { backgroundColor: '#000', borderColor: '#000' },
  categoryCheckboxText: { fontSize: 15, color: '#666' },
  categoryCheckboxTextActive: { color: '#000', fontWeight: '600' },
  applyBtn: { paddingVertical: 14, backgroundColor: '#000', borderRadius: 10, alignItems: 'center' },
  applyBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },

  // Date modal
  dateModal: { backgroundColor: '#fff', borderRadius: 16, padding: 20, width: 320 },
  dateModalTitle: { fontSize: 17, fontWeight: '700', color: '#000', textAlign: 'center', marginBottom: 14 },
  todayOption: { paddingVertical: 10, paddingHorizontal: 16, borderRadius: 10, borderWidth: 1, borderColor: '#e0e0e0', alignItems: 'center', marginBottom: 14 },
  todayOptionActive: { backgroundColor: '#000', borderColor: '#000' },
  todayOptionText: { fontSize: 15, fontWeight: '600', color: '#444' },
  todayOptionTextActive: { color: '#fff' },
  yearNav: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginBottom: 12, gap: 20 },
  yearNavArrow: { padding: 4 },
  yearNavText: { fontSize: 18, fontWeight: '700', color: '#000', minWidth: 60, textAlign: 'center' },
  monthGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, justifyContent: 'space-between', marginBottom: 14 },
  monthCell: { width: '30%', paddingVertical: 9, borderRadius: 8, borderWidth: 1, borderColor: '#e0e0e0', alignItems: 'center' },
  monthCellActive: { backgroundColor: '#000', borderColor: '#000' },
  monthCellText: { fontSize: 13, color: '#555', fontWeight: '500' },
  monthCellTextActive: { color: '#fff', fontWeight: '700' },
  customRangeBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 10, paddingHorizontal: 14, borderRadius: 10, borderWidth: 1, borderColor: '#ddd', justifyContent: 'center' },
  customRangeBtnActive: { backgroundColor: '#000', borderColor: '#000' },
  customRangeBtnText: { fontSize: 13, color: '#666' },
  customRangeBtnTextActive: { color: '#fff' },
  backRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 20 },
  backRowText: { fontSize: 16, fontWeight: '700', color: '#000' },
  rangeLabel: { fontSize: 13, fontWeight: '600', color: '#999', marginBottom: 8 },
  dateRow: { flexDirection: 'row', gap: 8, justifyContent: 'center' },
  spinner: { alignItems: 'center', flex: 1 },
  spinnerLabel: { fontSize: 11, color: '#999', marginBottom: 4 },
  spinnerArrow: { padding: 4 },
  spinnerValue: { fontSize: 15, fontWeight: '700', color: '#000', minWidth: 44, textAlign: 'center', paddingVertical: 4 },

  // Likes
  likesList: { flex: 1, padding: 16 },
  emptyState: { alignItems: 'center', paddingTop: 100 },
  emptyText: { marginTop: 12, fontSize: 16, color: '#999' },
  likedCard: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#fafafa', borderRadius: 12, padding: 16, marginBottom: 12, borderWidth: 1, borderColor: '#f0f0f0' },
  likedContent: { flex: 1, marginRight: 12 },
  likedTitle: { fontSize: 15, fontWeight: '600', color: '#000', lineHeight: 20 },
  likedAuthors: { fontSize: 13, color: '#666', marginTop: 4 },
  likedDate: { fontSize: 12, color: '#999', marginTop: 4 },
  exportBtn: { marginHorizontal: 16, marginBottom: 16, paddingVertical: 14, backgroundColor: '#000', borderRadius: 12, alignItems: 'center' },
  exportBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },

  // Settings
  headerTitle: { fontSize: 17, fontWeight: '600', color: '#000' },
  settingsContent: { flex: 1, padding: 16 },
  divider: { height: 1, backgroundColor: '#f0f0f0', marginVertical: 24 },
  supportCard: { borderRadius: 14, backgroundColor: '#000', marginBottom: 8 },
  supportCardContent: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 18 },
  supportCardTitle: { fontSize: 16, fontWeight: '700', color: '#fff' },
  supportCardSub: { fontSize: 13, color: 'rgba(255,255,255,0.6)', marginTop: 2 },
  settingsItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: 16, borderBottomWidth: 1, borderBottomColor: '#f5f5f5' },
  settingsItemContent: { flex: 1, marginLeft: 12 },
  settingsItemTitle: { fontSize: 16, color: '#000' },
  settingsItemSub: { fontSize: 13, color: '#999', marginTop: 2 },

  // Login screen
  loginContainer: { flex: 1, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center', paddingHorizontal: 40 },
  loginTitle: { fontSize: 32, fontWeight: '700', color: '#000', marginBottom: 8 },
  loginSubtitle: { fontSize: 16, color: '#666', textAlign: 'center', marginBottom: 48 },
  googleBtn: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: '#000', paddingVertical: 14, paddingHorizontal: 28, borderRadius: 14 },
  googleBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  guestBtn: { marginTop: 20, fontSize: 14, color: '#999' },

  // Sign out
  signOutBtn: { marginTop: 24, paddingVertical: 14, borderRadius: 10, borderWidth: 1, borderColor: '#e0e0e0', alignItems: 'center' },
  signOutBtnText: { fontSize: 15, color: '#666' },

  // Onboarding hint
  onboardingOverlay: { position: 'absolute', bottom: 72, left: 0, right: 0, alignItems: 'center', pointerEvents: 'box-none' },
  onboardingCard: { backgroundColor: 'rgba(0,0,0,0.72)', borderRadius: 14, paddingVertical: 14, paddingHorizontal: 20, gap: 10, maxWidth: 290 },
  onboardingCategoryHint: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: '#fff', borderRadius: 8, paddingVertical: 7, paddingHorizontal: 12, marginBottom: 2 },
  onboardingCategoryText: { fontSize: 13, fontWeight: '700', color: '#000' },
  onboardingRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  onboardingText: { fontSize: 13, color: 'rgba(255,255,255,0.88)', letterSpacing: 0.1 },
});
