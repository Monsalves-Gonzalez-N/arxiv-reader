import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  ActivityIndicator,
  TouchableOpacity,
  StatusBar,
  Platform,
  ScrollView,
  Linking,
  Share,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { GestureHandlerRootView, Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  runOnJS,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
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

const YEARS = Array.from({ length: 36 }, (_, i) => 2025 - i); // 2025 to 1990

function cleanLatex(text: string): string {
  if (!text) return text;
  const replacements: [RegExp, string][] = [
    // Greek lowercase
    [/\\alpha/g,'α'],[/\\beta/g,'β'],[/\\gamma/g,'γ'],[/\\delta/g,'δ'],
    [/\\varepsilon/g,'ε'],[/\\epsilon/g,'ε'],[/\\zeta/g,'ζ'],[/\\eta/g,'η'],
    [/\\vartheta/g,'ϑ'],[/\\theta/g,'θ'],[/\\iota/g,'ι'],[/\\kappa/g,'κ'],
    [/\\lambda/g,'λ'],[/\\mu/g,'μ'],[/\\nu/g,'ν'],[/\\xi/g,'ξ'],
    [/\\varpi/g,'ϖ'],[/\\pi/g,'π'],[/\\varrho/g,'ϱ'],[/\\rho/g,'ρ'],
    [/\\varsigma/g,'ς'],[/\\sigma/g,'σ'],[/\\tau/g,'τ'],[/\\upsilon/g,'υ'],
    [/\\varphi/g,'φ'],[/\\phi/g,'φ'],[/\\chi/g,'χ'],[/\\psi/g,'ψ'],[/\\omega/g,'ω'],
    // Greek uppercase
    [/\\Gamma/g,'Γ'],[/\\Delta/g,'Δ'],[/\\Theta/g,'Θ'],[/\\Lambda/g,'Λ'],
    [/\\Xi/g,'Ξ'],[/\\Pi/g,'Π'],[/\\Sigma/g,'Σ'],[/\\Upsilon/g,'Υ'],
    [/\\Phi/g,'Φ'],[/\\Psi/g,'Ψ'],[/\\Omega/g,'Ω'],
    // Math symbols
    [/\\odot/g,'⊙'],[/\\oplus/g,'⊕'],[/\\otimes/g,'⊗'],[/\\cdots/g,'⋯'],
    [/\\ldots/g,'…'],[/\\cdot/g,'·'],[/\\times/g,'×'],[/\\div/g,'÷'],
    [/\\pm/g,'±'],[/\\mp/g,'∓'],[/\\leq/g,'≤'],[/\\geq/g,'≥'],
    [/\\neq/g,'≠'],[/\\lesssim/g,'≲'],[/\\gtrsim/g,'≳'],[/\\ll/g,'≪'],
    [/\\gg/g,'≫'],[/\\approx/g,'≈'],[/\\simeq/g,'≃'],[/\\equiv/g,'≡'],
    [/\\sim/g,'~'],[/\\propto/g,'∝'],[/\\infty/g,'∞'],[/\\partial/g,'∂'],
    [/\\nabla/g,'∇'],[/\\rightarrow/g,'→'],[/\\leftarrow/g,'←'],
    [/\\leftrightarrow/g,'↔'],[/\\Rightarrow/g,'⇒'],[/\\to/g,'→'],
    [/\\in\b/g,'∈'],[/\\notin/g,'∉'],[/\\subset/g,'⊂'],[/\\supset/g,'⊃'],
    [/\\cup/g,'∪'],[/\\cap/g,'∩'],[/\\forall/g,'∀'],[/\\exists/g,'∃'],
    // sqrt
    [/\\sqrt\{([^}]+)\}/g,'√($1)'],[/\\sqrt/g,'√'],
    // Text formatting — extract content
    [/\\(?:text|mathrm|mathbf|mathit|mathcal|textbf|textit|emph)\{([^}]+)\}/g,'$1'],
    // sub/superscript
    [/\^\{([^}]+)\}/g,'^($1)'],[/_\{([^}]+)\}/g,'_($1)'],
    [/\^([A-Za-z0-9])/g,'^$1'],[/_([A-Za-z0-9])/g,'_$1'],
    // Remove $ markers
    [/\$\$/g,''],[/\$/g,''],
    // Remaining commands — strip braces, keep content
    [/\\[a-zA-Z]+\{([^}]*)\}/g,'$1'],[/\\[a-zA-Z]+/g,''],
    [/[{}]/g,''],[/\s+/g,' '],
  ];
  let result = text;
  for (const [pattern, replacement] of replacements) result = result.replace(pattern, replacement);
  return result.trim();
}

function splitAbstract(abstract: string): string[] {
  if (!abstract) return [''];
  const sentences = abstract.match(/[^.!?]+[.!?]+/g) || [abstract];
  const parts: string[] = [];
  let currentPart = '';
  
  for (const sentence of sentences) {
    const trimmedSentence = sentence.trim();
    if (currentPart.length + trimmedSentence.length <= MAX_CHARS_PER_PART) {
      currentPart += (currentPart ? ' ' : '') + trimmedSentence;
    } else {
      if (currentPart) parts.push(currentPart);
      currentPart = trimmedSentence;
    }
  }
  if (currentPart) parts.push(currentPart);
  return parts.length > 0 ? parts : [''];
}

export default function Index() {
  const [deviceId, setDeviceId] = useState<string>('');
  const seenIds = useRef<Set<string>>(new Set());
  const [refreshing, setRefreshing] = useState(false);
  const [allPapersSeen, setAllPapersSeen] = useState(false);

  const [papers, setPapers] = useState<Paper[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set([]));
  const [likedPapers, setLikedPapers] = useState<Set<string>>(new Set());
  const [likedPapersList, setLikedPapersList] = useState<any[]>([]);
  const [hasMore, setHasMore] = useState(true);
  
  // For You state
  const [forYouPapers, setForYouPapers] = useState<Paper[]>([]);
  const [forYouIndex, setForYouIndex] = useState(0);
  const [yearFrom, setYearFrom] = useState(2024);
  const [yearTo, setYearTo] = useState(2026);
  const [forYouLoading, setForYouLoading] = useState(false);
  
  const [currentView, setCurrentView] = useState<'feed' | 'foryou' | 'likes' | 'settings'>('feed');
  const [abstractPart, setAbstractPart] = useState(0);
  const [showCategoryPicker, setShowCategoryPicker] = useState(false);
  const [showYearPicker, setShowYearPicker] = useState<'from' | 'to' | null>(null);
  const [todayOnly, setTodayOnly] = useState(false);

  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);

  const displayPapers = useMemo(() =>
    todayOnly ? papers.filter(p => p.is_new) : papers,
    [papers, todayOnly]
  );

  const currentPaper = currentView === 'foryou' ? forYouPapers[forYouIndex] : displayPapers[currentIndex];
  const abstractParts = useMemo(() => 
    currentPaper ? splitAbstract(cleanLatex(currentPaper.abstract)) : [''],
    [currentPaper?.abstract]
  );
  const totalParts = abstractParts.length + 1;

  useEffect(() => {
    initDeviceId();
    loadSavedCategories();
  }, []);

  const initDeviceId = async () => {
    let id = await AsyncStorage.getItem('device_id');
    if (!id) {
      id = `device_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
      await AsyncStorage.setItem('device_id', id);
    }
    setDeviceId(id);
  };

  const loadSavedCategories = async () => {
    try {
      const saved = await AsyncStorage.getItem('selected_categories');
      if (saved) {
        setSelectedCategories(new Set(JSON.parse(saved)));
      } else {
        setSelectedCategories(new Set(['astro-ph.GA', 'astro-ph.CO']));
      }
    } catch {
      setSelectedCategories(new Set(['astro-ph.GA', 'astro-ph.CO']));
    }
  };

  const fetchPapers = useCallback(async (categories: Set<string>, start: number = 0, append: boolean = false) => {
    try {
      if (start === 0 && !append) setLoading(true);
      else setLoadingMore(true);

      const id = await AsyncStorage.getItem('device_id');
      const headers: any = id ? { 'X-Device-ID': id } : {};

      const categoryArray = Array.from(categories);
      const category = categoryArray.join(',');

      const response = await fetch(
        `${EXPO_PUBLIC_BACKEND_URL}/api/papers/feed?category=${category}&start=${start}&max_results=20`,
        { headers }
      );
      const data = await response.json();
      const fetched: Paper[] = data.papers || [];

      const fresh = fetched.filter(p => !seenIds.current.has(p.id));
      fresh.forEach(p => seenIds.current.add(p.id));

      if (fresh.length === 0 && !append) {
        setAllPapersSeen(true);
      } else {
        setAllPapersSeen(false);
        if (append) {
          setPapers(prev => [...prev, ...fresh]);
        } else {
          setPapers(fresh);
          setCurrentIndex(0);
          setAbstractPart(0);
        }
      }
      setHasMore(data.has_more ?? false);
    } catch (error) {
      console.error('Error fetching papers:', error);
    } finally {
      setLoading(false);
      setLoadingMore(false);
      setRefreshing(false);
    }
  }, []);

  const fetchForYouPapers = useCallback(async () => {
    try {
      setForYouLoading(true);

      const id = await AsyncStorage.getItem('device_id');
      const headers: any = id ? { 'X-Device-ID': id } : {};
      
      const response = await fetch(
        `${EXPO_PUBLIC_BACKEND_URL}/api/for-you?category=astro-ph&year_from=${yearFrom}&year_to=${yearTo}&max_results=30`,
        { headers }
      );
      const data = await response.json();
      
      setForYouPapers(data.papers || []);
      setForYouIndex(0);
      setAbstractPart(0);
    } catch (error) {
      console.error('Error fetching for-you papers:', error);
    } finally {
      setForYouLoading(false);
    }
  }, [yearFrom, yearTo]);

  const fetchLikedPapers = useCallback(async () => {
    try {
      const id = await AsyncStorage.getItem('device_id');
      const headers: any = id ? { 'X-Device-ID': id } : {};

      const response = await fetch(`${EXPO_PUBLIC_BACKEND_URL}/api/likes`, { headers });
      const data = await response.json();
      setLikedPapers(new Set(data.map((p: any) => p.paper_id)));
      setLikedPapersList(data);
    } catch (error) {
      console.error('Error fetching liked papers:', error);
    }
  }, []);

  useEffect(() => {
    if (selectedCategories.size > 0) {
      fetchPapers(selectedCategories);
      fetchLikedPapers();
    }
  }, [selectedCategories]);

  useEffect(() => {
    if (currentView === 'foryou') {
      fetchForYouPapers();
    }
  }, [currentView, yearFrom, yearTo]);

  const toggleCategory = (categoryKey: string) => {
    seenIds.current.clear();
    setAllPapersSeen(false);
    setSelectedCategories(prev => {
      const newSet = new Set(prev);
      if (newSet.has(categoryKey)) {
        if (newSet.size > 1) newSet.delete(categoryKey);
      } else {
        newSet.add(categoryKey);
      }
      AsyncStorage.setItem('selected_categories', JSON.stringify(Array.from(newSet)));
      return newSet;
    });
  };

  const toggleLike = async (paper: Paper) => {
    const isLiked = likedPapers.has(paper.id);
    const id = await AsyncStorage.getItem('device_id');
    const headers: any = { 'Content-Type': 'application/json' };
    if (id) headers['X-Device-ID'] = id;
    
    try {
      if (isLiked) {
        await fetch(`${EXPO_PUBLIC_BACKEND_URL}/api/likes/${paper.id}`, {
          method: 'DELETE',
          headers,
        });
        setLikedPapers(prev => {
          const newSet = new Set(prev);
          newSet.delete(paper.id);
          return newSet;
        });
      } else {
        await fetch(`${EXPO_PUBLIC_BACKEND_URL}/api/likes`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            paper_id: paper.id,
            title: paper.title,
            abstract: paper.abstract,
            authors: paper.authors,
            published: paper.published,
            link: paper.link,
            category: paper.category,
          }),
        });
        setLikedPapers(prev => new Set(prev).add(paper.id));
      }
      fetchLikedPapers();
    } catch (error) {
      console.error('Error toggling like:', error);
    }
  };

  // Mark paper as viewed (to avoid repetition like TikTok)
  const markPaperViewed = async (paperId: string) => {
    try {
      const id = await AsyncStorage.getItem('device_id');
      const headers: any = { 'Content-Type': 'application/json' };
      if (id) headers['X-Device-ID'] = id;
      
      await fetch(`${EXPO_PUBLIC_BACKEND_URL}/api/papers/mark-viewed/${paperId}`, {
        method: 'POST',
        headers,
      });
    } catch (error) {
      // Silent fail - not critical
    }
  };

  const goToNext = useCallback(() => {
    // Mark current paper as viewed before moving to next
    const paper = currentView === 'foryou' ? forYouPapers[forYouIndex] : displayPapers[currentIndex];
    if (paper) {
      markPaperViewed(paper.id);
    }

    if (currentView === 'foryou') {
      if (forYouIndex < forYouPapers.length - 1) {
        setForYouIndex(prev => prev + 1);
        setAbstractPart(0);
      }
    } else {
      if (currentIndex < displayPapers.length - 1) {
        setCurrentIndex(prev => prev + 1);
        setAbstractPart(0);
        if (!todayOnly && currentIndex >= papers.length - 5 && hasMore && !loadingMore) {
          fetchPapers(selectedCategories, papers.length, true);
        }
      } else if (!loadingMore && !refreshing) {
        // At the last paper — try fetching new papers
        setRefreshing(true);
        fetchPapers(selectedCategories, 0, false);
      }
    }
  }, [currentView, currentIndex, forYouIndex, displayPapers, papers, forYouPapers, hasMore, loadingMore, selectedCategories, fetchPapers, todayOnly]);

  const goToPrevious = useCallback(() => {
    if (currentView === 'foryou') {
      if (forYouIndex > 0) {
        setForYouIndex(prev => prev - 1);
        setAbstractPart(0);
      }
    } else {
      if (currentIndex > 0) {
        setCurrentIndex(prev => prev - 1);
        setAbstractPart(0);
      }
    }
  }, [currentView, currentIndex, forYouIndex]);

  const nextAbstractPart = useCallback(() => {
    if (abstractPart < abstractParts.length) {
      setAbstractPart(prev => prev + 1);
    }
  }, [abstractPart, abstractParts.length]);

  const prevAbstractPart = useCallback(() => {
    if (abstractPart > 0) {
      setAbstractPart(prev => prev - 1);
    }
  }, [abstractPart]);

  const panGesture = Gesture.Pan()
    .minDistance(10)
    .onUpdate((e) => {
      translateX.value = e.translationX;
      translateY.value = e.translationY;
    })
    .onEnd((e) => {
      const { translationX: tx, translationY: ty, velocityX: vx, velocityY: vy } = e;
      
      // Vertical swipe always changes paper (even in abstract)
      if (Math.abs(ty) > Math.abs(tx) * 0.8) {
        if (ty < -SWIPE_THRESHOLD || vy < -500) {
          runOnJS(goToNext)();
        } else if (ty > SWIPE_THRESHOLD || vy > 500) {
          runOnJS(goToPrevious)();
        }
      } 
      // Horizontal swipe changes abstract part
      else if (Math.abs(tx) > SWIPE_THRESHOLD || Math.abs(vx) > 500) {
        if (tx < -SWIPE_THRESHOLD || vx < -500) {
          runOnJS(nextAbstractPart)();
        } else if (tx > SWIPE_THRESHOLD || vx > 500) {
          runOnJS(prevAbstractPart)();
        }
      }
      
      translateX.value = 0;
      translateY.value = 0;
    });

  const animatedCardStyle = useAnimatedStyle(() => ({
    transform: [],
  }));

  const exportToCSV = async () => {
    if (likedPapersList.length === 0) return;
    const headers = 'Title,Authors,Published,Link\n';
    const rows = likedPapersList.map(p =>
      `"${p.title.replace(/"/g, '""')}","${p.authors.join('; ')}",${p.published},${p.link}`
    ).join('\n');
    await Share.share({ message: headers + rows, title: 'My arXiv Papers' });
  };

  const openCoffee = () => Linking.openURL(COFFEE_LINK);
  const openPaper = () => currentPaper && Linking.openURL(currentPaper.link);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <StatusBar barStyle="dark-content" />
        <ActivityIndicator size="large" color="#000" />
        <Text style={styles.loadingText}>Loading...</Text>
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
            ) : (
              likedPapersList.map((paper, idx) => (
                <TouchableOpacity
                  key={paper.paper_id || idx}
                  style={styles.likedCard}
                  onPress={() => Linking.openURL(paper.link)}
                >
                  <View style={styles.likedContent}>
                    <Text style={styles.likedTitle} numberOfLines={2}>{paper.title}</Text>
                    <Text style={styles.likedAuthors} numberOfLines={1}>
                      {paper.authors.slice(0, 2).join(', ')}
                    </Text>
                    <Text style={styles.likedDate}>{paper.published}</Text>
                  </View>
                  <TouchableOpacity onPress={() => toggleLike(paper)}>
                    <Ionicons name="heart" size={20} color="#000" />
                  </TouchableOpacity>
                </TouchableOpacity>
              ))
            )}
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

            <TouchableOpacity 
              style={styles.settingsItem} 
              onPress={() => Linking.openURL('https://arxiv.org')}
            >
              <Ionicons name="globe-outline" size={24} color="#000" />
              <View style={styles.settingsItemContent}>
                <Text style={styles.settingsItemTitle}>arXiv.org</Text>
                <Text style={styles.settingsItemSub}>Data source</Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#ccc" />
            </TouchableOpacity>
          </ScrollView>
        </SafeAreaView>
      </GestureHandlerRootView>
    );
  }

  // For You View
  if (currentView === 'foryou') {
    return (
      <GestureHandlerRootView style={styles.container}>
        <StatusBar barStyle="dark-content" />
        <SafeAreaView style={styles.safeArea}>
          <View style={styles.header}>
            <TouchableOpacity onPress={() => setCurrentView('feed')}>
              <Ionicons name="arrow-back" size={24} color="#000" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>For You</Text>
            <View style={{ width: 24 }} />
          </View>

          {/* Year Selector */}
          <View style={styles.yearSelector}>
            <TouchableOpacity 
              style={styles.yearBtn}
              onPress={() => setShowYearPicker('from')}
            >
              <Text style={styles.yearLabel}>From</Text>
              <Text style={styles.yearValue}>{yearFrom}</Text>
            </TouchableOpacity>
            <Text style={styles.yearDash}>—</Text>
            <TouchableOpacity 
              style={styles.yearBtn}
              onPress={() => setShowYearPicker('to')}
            >
              <Text style={styles.yearLabel}>To</Text>
              <Text style={styles.yearValue}>{yearTo}</Text>
            </TouchableOpacity>
          </View>

          {forYouLoading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#000" />
              <Text style={styles.loadingText}>Finding papers for you...</Text>
            </View>
          ) : (
            <GestureDetector gesture={panGesture}>
              <Animated.View style={[styles.cardArea, animatedCardStyle]}>
                {currentPaper ? (
                  <View style={styles.card}>
                    {currentPaper.is_recommendation ? (
                      <View style={styles.recBadge}>
                        <Text style={styles.recBadgeText}>Recommended</Text>
                      </View>
                    ) : (
                      <View style={[styles.recBadge, { backgroundColor: '#e8f4e8' }]}>
                        <Text style={[styles.recBadgeText, { color: '#4a7c4a' }]}>Explore</Text>
                      </View>
                    )}

                    <View style={styles.cardContent}>
                      {abstractPart === 0 ? (
                        <>
                          <Text style={styles.paperTitle}>{cleanLatex(currentPaper.title)}</Text>
                          {currentPaper.comment && (
                            <View style={styles.commentBadge}>
                              <Text style={styles.commentText} numberOfLines={2}>
                                {currentPaper.comment}
                              </Text>
                            </View>
                          )}
                          <Text style={styles.paperDate}>{currentPaper.published}</Text>
                          <TouchableOpacity style={styles.linkBtn} onPress={openPaper}>
                            <Ionicons name="open-outline" size={16} color="#666" />
                            <Text style={styles.linkBtnText}>Open in arXiv</Text>
                          </TouchableOpacity>
                        </>
                      ) : (
                        <>
                          <Text style={styles.abstractPartLabel}>
                            Part {abstractPart}/{abstractParts.length}
                          </Text>
                          <Text style={styles.abstractText}>
                            {abstractParts[abstractPart - 1] || ''}
                          </Text>
                        </>
                      )}
                    </View>

                    <View style={styles.dots}>
                      {Array.from({ length: totalParts }, (_, i) => (
                        <View key={i} style={[styles.dot, abstractPart === i && styles.dotActive]} />
                      ))}
                    </View>

                    <Text style={styles.counter}>
                      {forYouIndex + 1} / {forYouPapers.length}
                    </Text>
                  </View>
                ) : (
                  <View style={styles.emptyCard}>
                    <Ionicons name="sparkles-outline" size={48} color="#ccc" />
                    <Text style={styles.emptyCardText}>No papers found for this year range</Text>
                    <Text style={styles.emptyCardSubtext}>Try adjusting the years above</Text>
                  </View>
                )}
              </Animated.View>
            </GestureDetector>
          )}

          <View style={styles.bottomBar}>
            <View style={styles.swipeHints}>
              <Text style={styles.hintText}>
                {abstractPart === 0 ? `Swipe ← for abstract` : 'Swipe → to go back'}
              </Text>
            </View>
            
            {currentPaper && (
              <TouchableOpacity
                style={styles.likeBtn}
                onPress={() => toggleLike(currentPaper)}
              >
                <Ionicons
                  name={likedPapers.has(currentPaper.id) ? 'heart' : 'heart-outline'}
                  size={28}
                  color={likedPapers.has(currentPaper.id) ? '#000' : '#666'}
                />
              </TouchableOpacity>
            )}
          </View>

          {/* Year Picker Modal */}
          {showYearPicker && (
            <TouchableOpacity
              style={styles.modalOverlay}
              activeOpacity={1}
              onPress={() => setShowYearPicker(null)}
            >
              <View style={styles.yearModal}>
                <Text style={styles.yearModalTitle}>
                  Select {showYearPicker === 'from' ? 'Start' : 'End'} Year
                </Text>
                <ScrollView style={styles.yearList}>
                  {YEARS.map(year => (
                    <TouchableOpacity
                      key={year}
                      style={[
                        styles.yearItem,
                        (showYearPicker === 'from' ? yearFrom : yearTo) === year && styles.yearItemActive
                      ]}
                      onPress={() => {
                        if (showYearPicker === 'from') setYearFrom(year);
                        else setYearTo(year);
                        setShowYearPicker(null);
                      }}
                    >
                      <Text style={[
                        styles.yearItemText,
                        (showYearPicker === 'from' ? yearFrom : yearTo) === year && styles.yearItemTextActive
                      ]}>
                        {year}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
              </View>
            </TouchableOpacity>
          )}
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
          <TouchableOpacity
            style={styles.categoryBtn}
            onPress={() => setShowCategoryPicker(true)}
          >
            <Text style={styles.categoryBtnText}>
              {selectedCategories.size === CATEGORIES.length 
                ? 'All' 
                : `${selectedCategories.size} selected`}
            </Text>
            <Ionicons name="chevron-down" size={16} color="#000" />
          </TouchableOpacity>

          <View style={styles.headerRight}>
            <TouchableOpacity
              style={[styles.todayBtn, todayOnly && styles.todayBtnActive]}
              onPress={() => { setTodayOnly(prev => !prev); setCurrentIndex(0); setAbstractPart(0); }}
            >
              <Text style={[styles.todayBtnText, todayOnly && styles.todayBtnTextActive]}>Today</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.iconBtn}
              onPress={() => setCurrentView('foryou')}
            >
              <Ionicons name="sparkles" size={22} color="#000" />
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.iconBtn}
              onPress={() => { fetchLikedPapers(); setCurrentView('likes'); }}
            >
              <Ionicons name="heart" size={22} color="#000" />
              {likedPapers.size > 0 && (
                <View style={styles.badge}>
                  <Text style={styles.badgeText}>{likedPapers.size}</Text>
                </View>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.iconBtn}
              onPress={() => setCurrentView('settings')}
            >
              <Ionicons name="person-outline" size={22} color="#000" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Main Card */}
        <GestureDetector gesture={panGesture}>
          <Animated.View style={[styles.cardArea, animatedCardStyle]}>
            {allPapersSeen || (!currentPaper && !loading) ? (
              <View style={styles.emptyCard}>
                <Ionicons name="checkmark-circle-outline" size={48} color="#ccc" />
                <Text style={styles.emptyCardText}>
                  {todayOnly ? "You've seen all of today's papers" : "You've seen all available papers"}
                </Text>
                <Text style={styles.emptyCardSubtext}>Check back tomorrow for new listings</Text>
              </View>
            ) : currentPaper && (
              <View style={styles.card}>
                {currentPaper.is_new && (
                  <View style={styles.newBadge}>
                    <Text style={styles.newBadgeText}>NEW</Text>
                  </View>
                )}

                {currentPaper.is_recommendation && (
                  <View style={styles.recBadge}>
                    <Text style={styles.recBadgeText}>For You</Text>
                  </View>
                )}

                <View style={styles.cardContent}>
                  {abstractPart === 0 ? (
                    <>
                      <Text style={styles.paperTitle}>{cleanLatex(currentPaper.title)}</Text>
                      {currentPaper.comment && (
                        <View style={styles.commentBadge}>
                          <Text style={styles.commentText} numberOfLines={2}>
                            {currentPaper.comment}
                          </Text>
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
                      <Text style={styles.abstractPartLabel}>
                        Part {abstractPart}/{abstractParts.length}
                      </Text>
                      <Text style={styles.abstractText} numberOfLines={12}>
                        {abstractParts[abstractPart - 1] || ''}
                      </Text>
                    </View>
                  )}
                </View>

                <View style={styles.dots}>
                  {Array.from({ length: totalParts }, (_, i) => (
                    <View key={i} style={[styles.dot, abstractPart === i && styles.dotActive]} />
                  ))}
                </View>

                <Text style={styles.counter}>
                  {currentIndex + 1} / {displayPapers.length}
                </Text>
              </View>
            )}
          </Animated.View>
        </GestureDetector>

        {/* Bottom Actions */}

        <View style={styles.bottomBar}>
          <View style={styles.swipeHints}>
            <Text style={styles.hintText}>
              {abstractPart === 0
                ? currentIndex === displayPapers.length - 1
                  ? 'Swipe ↑ to refresh'
                  : `Swipe ← for abstract (${abstractParts.length} parts)`
                : 'Swipe → to go back'}
            </Text>
          </View>
          
          <TouchableOpacity
            style={styles.likeBtn}
            onPress={() => currentPaper && toggleLike(currentPaper)}
          >
            <Ionicons
              name={currentPaper && likedPapers.has(currentPaper.id) ? 'heart' : 'heart-outline'}
              size={28}
              color={currentPaper && likedPapers.has(currentPaper.id) ? '#000' : '#666'}
            />
          </TouchableOpacity>
        </View>

        {/* Category Picker Modal */}
        {showCategoryPicker && (
          <TouchableOpacity
            style={styles.modalOverlay}
            activeOpacity={1}
            onPress={() => {
              setShowCategoryPicker(false);
              fetchPapers(selectedCategories);
            }}
          >
            <View style={styles.categoryModal}>
              <Text style={styles.categoryModalTitle}>Select Categories</Text>
              <Text style={styles.categoryModalSub}>Tap to select multiple</Text>
              
              <View style={styles.categoryGrid}>
                {CATEGORIES.map(cat => (
                  <TouchableOpacity
                    key={cat.key}
                    style={[
                      styles.categoryCheckbox,
                      selectedCategories.has(cat.key) && styles.categoryCheckboxActive
                    ]}
                    onPress={() => toggleCategory(cat.key)}
                  >
                    <View style={[
                      styles.checkbox,
                      selectedCategories.has(cat.key) && styles.checkboxActive
                    ]}>
                      {selectedCategories.has(cat.key) && (
                        <Ionicons name="checkmark" size={14} color="#fff" />
                      )}
                    </View>
                    <Text style={[
                      styles.categoryCheckboxText,
                      selectedCategories.has(cat.key) && styles.categoryCheckboxTextActive
                    ]}>
                      {cat.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>

              <TouchableOpacity 
                style={styles.applyBtn}
                onPress={() => {
                  setShowCategoryPicker(false);
                  fetchPapers(selectedCategories);
                }}
              >
                <Text style={styles.applyBtnText}>Apply</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        )}

        {(loadingMore || refreshing) && (
          <View style={styles.loadingMore}>
            <ActivityIndicator size="small" color="#000" />
            {refreshing && <Text style={styles.refreshingText}>Updating papers...</Text>}
          </View>
        )}
      </SafeAreaView>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  safeArea: { flex: 1 },
  loadingContainer: { flex: 1, backgroundColor: '#fff', justifyContent: 'center', alignItems: 'center' },
  loadingText: { marginTop: 12, color: '#999', fontSize: 14 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 20, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#f0f0f0' },
  headerTitle: { fontSize: 17, fontWeight: '600', color: '#000' },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  categoryBtn: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  categoryBtnText: { fontSize: 17, fontWeight: '600', color: '#000' },
  iconBtn: { position: 'relative', padding: 4 },
  todayBtn: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 12, borderWidth: 1, borderColor: '#ccc' },
  todayBtnActive: { backgroundColor: '#000', borderColor: '#000' },
  todayBtnText: { fontSize: 13, fontWeight: '600', color: '#666' },
  todayBtnTextActive: { color: '#fff' },
  badge: { position: 'absolute', top: -2, right: -2, backgroundColor: '#000', borderRadius: 8, minWidth: 16, height: 16, justifyContent: 'center', alignItems: 'center' },
  badgeText: { color: '#fff', fontSize: 10, fontWeight: '600' },
  cardArea: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  card: { width: '100%', height: '100%', backgroundColor: '#fafafa', borderRadius: 16, padding: 24, justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: '#e0e0e0', overflow: 'hidden' },
  emptyCard: { width: '100%', height: '100%', backgroundColor: '#fafafa', borderRadius: 16, padding: 24, justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: '#e0e0e0', overflow: 'hidden' },
  emptyCardText: { marginTop: 16, color: '#999', fontSize: 16, textAlign: 'center' },
  emptyCardSubtext: { marginTop: 8, color: '#ccc', fontSize: 14, textAlign: 'center' },
  newBadge: { position: 'absolute', top: 16, left: 16, backgroundColor: '#000', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4 },
  newBadgeText: { color: '#fff', fontSize: 10, fontWeight: '700' },
  recBadge: { position: 'absolute', top: 16, right: 16, backgroundColor: '#f0f0f0', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4 },
  recBadgeText: { color: '#666', fontSize: 10, fontWeight: '600' },
  cardContent: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 20, width: '100%' },
  paperTitle: { fontSize: 20, fontWeight: '700', color: '#000', textAlign: 'center', lineHeight: 28 },
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
  bottomBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 24, paddingVertical: 16, borderTopWidth: 1, borderTopColor: '#f0f0f0' },
  swipeHints: { flex: 1 },
  hintText: { fontSize: 13, color: '#999' },
  likeBtn: { width: 48, height: 48, borderRadius: 24, backgroundColor: '#f5f5f5', justifyContent: 'center', alignItems: 'center' },
  modalOverlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', alignItems: 'center' },
  categoryModal: { backgroundColor: '#fff', borderRadius: 16, padding: 24, width: '85%', maxWidth: 340 },
  categoryModalTitle: { fontSize: 18, fontWeight: '700', color: '#000', textAlign: 'center' },
  categoryModalSub: { fontSize: 13, color: '#999', textAlign: 'center', marginTop: 4, marginBottom: 20 },
  categoryGrid: { gap: 10 },
  categoryCheckbox: { flexDirection: 'row', alignItems: 'center', paddingVertical: 12, paddingHorizontal: 12, borderRadius: 8, borderWidth: 1, borderColor: '#e0e0e0', gap: 12 },
  categoryCheckboxActive: { borderColor: '#000', backgroundColor: '#fafafa' },
  checkbox: { width: 22, height: 22, borderRadius: 4, borderWidth: 2, borderColor: '#ccc', justifyContent: 'center', alignItems: 'center' },
  checkboxActive: { backgroundColor: '#000', borderColor: '#000' },
  categoryCheckboxText: { fontSize: 15, color: '#666' },
  categoryCheckboxTextActive: { color: '#000', fontWeight: '600' },
  applyBtn: { marginTop: 20, paddingVertical: 14, backgroundColor: '#000', borderRadius: 10, alignItems: 'center' },
  applyBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  loadingMore: { position: 'absolute', bottom: 100, alignSelf: 'center', alignItems: 'center', gap: 4 },
  refreshingText: { fontSize: 12, color: '#666', marginTop: 4 },
  likesList: { flex: 1, padding: 16 },
  emptyState: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingTop: 100 },
  emptyText: { marginTop: 12, fontSize: 16, color: '#999' },
  likedCard: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#fafafa', borderRadius: 12, padding: 16, marginBottom: 12, borderWidth: 1, borderColor: '#f0f0f0' },
  likedContent: { flex: 1, marginRight: 12 },
  likedTitle: { fontSize: 15, fontWeight: '600', color: '#000', lineHeight: 20 },
  likedAuthors: { fontSize: 13, color: '#666', marginTop: 4 },
  likedDate: { fontSize: 12, color: '#999', marginTop: 4 },
  exportBtn: { marginHorizontal: 16, marginBottom: 16, paddingVertical: 14, backgroundColor: '#000', borderRadius: 12, alignItems: 'center' },
  exportBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  settingsContent: { flex: 1, padding: 16 },
  userCard: { alignItems: 'center', paddingVertical: 24 },
  userPicture: { width: 80, height: 80, borderRadius: 40, marginBottom: 12 },
  userName: { fontSize: 20, fontWeight: '600', color: '#000' },
  userEmail: { fontSize: 14, color: '#666', marginTop: 4 },
  logoutBtn: { marginTop: 16, paddingHorizontal: 24, paddingVertical: 10, backgroundColor: '#f5f5f5', borderRadius: 8 },
  logoutBtnText: { fontSize: 14, color: '#666' },
  loginBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 14, backgroundColor: '#000', borderRadius: 12 },
  loginBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  divider: { height: 1, backgroundColor: '#f0f0f0', marginVertical: 24 },
  supportCard: { borderRadius: 14, backgroundColor: '#000', marginBottom: 8 },
  supportCardContent: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 18 },
  supportCardTitle: { fontSize: 16, fontWeight: '700', color: '#fff' },
  supportCardSub: { fontSize: 13, color: 'rgba(255,255,255,0.6)', marginTop: 2 },
  settingsItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: 16, borderBottomWidth: 1, borderBottomColor: '#f5f5f5' },
  settingsItemContent: { flex: 1, marginLeft: 12 },
  settingsItemTitle: { fontSize: 16, color: '#000' },
  settingsItemSub: { fontSize: 13, color: '#999', marginTop: 2 },
  // Year selector
  yearSelector: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, gap: 16, borderBottomWidth: 1, borderBottomColor: '#f0f0f0' },
  yearBtn: { alignItems: 'center', paddingHorizontal: 20, paddingVertical: 10, backgroundColor: '#f5f5f5', borderRadius: 10 },
  yearLabel: { fontSize: 11, color: '#999', marginBottom: 2 },
  yearValue: { fontSize: 18, fontWeight: '700', color: '#000' },
  yearDash: { fontSize: 20, color: '#ccc' },
  yearModal: { backgroundColor: '#fff', borderRadius: 16, padding: 20, width: '70%', maxWidth: 280, maxHeight: '60%' },
  yearModalTitle: { fontSize: 17, fontWeight: '600', color: '#000', textAlign: 'center', marginBottom: 16 },
  yearList: { maxHeight: 300 },
  yearItem: { paddingVertical: 12, paddingHorizontal: 16, borderRadius: 8 },
  yearItemActive: { backgroundColor: '#f5f5f5' },
  yearItemText: { fontSize: 16, color: '#666', textAlign: 'center' },
  yearItemTextActive: { color: '#000', fontWeight: '600' },
});
